import torch
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd
from sklearn.model_selection import KFold
from Dataset import get_combined_dataset, get_trainloader_single, get_dataloaders_trans
from Models import get_trans_model
from Utils import loglikelihood_trans
from tqdm import tqdm
from datetime import datetime
from Main import train_model, evaluate_model
from Consts import TRAIN_ROOT, BATCH, EPOCH, LR, NUM_CLASS, SEQ_L, SUBJECT

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nested_cv(seq_len):
    subject = CV1_SUBJECT
    datasets = ['1', '2', '3']
    root = TRAIN_ROOT
    batch_size = BATCH
    learning_rate = LR
    num_classes = NUM_CLASS  # Adjust based on actual classes
    seq_len = seq_len
    
    # Create result directories
    subject_results_root = f"without_temp_enc/{subject}/evaluation_results_{seq_len}"
    if not os.path.exists(subject_results_root):
        os.makedirs(subject_results_root)
    
    # Create Excel file
    excel_filename = f"{subject_results_root}/lr{learning_rate}_seq{seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if not os.path.exists(os.path.dirname(excel_filename)):
        os.makedirs(os.path.dirname(excel_filename))
    df = pd.DataFrame(columns=['Epoch', 'Fold', 'Loss', 'Accuracy', 'Seq_Len', 'Test_Index'])
    df.to_excel(excel_filename, index=False)
    
    # Combine datasets
    combined_data = []
    for dataset in datasets:
        data_mat = os.path.join(root, subject, f"{dataset}.mat")
        combined_data.append(get_combined_dataset(data_mat, seq_len))
    
    combined_data = torch.utils.data.ConcatDataset(combined_data)
    
    outer_kf = KFold(n_splits=5)
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_kf.split(combined_data)):
        train_val_data = torch.utils.data.Subset(combined_data, train_val_idx)
        test_data = torch.utils.data.Subset(combined_data, test_idx)
        
        inner_kf = KFold(n_splits=5)
        best_inner_accuracy = 0
        best_inner_epoch = 0
        
        for inner_fold, (train_idx, val_idx) in enumerate(inner_kf.split(train_val_data)):
            train_data = torch.utils.data.Subset(train_val_data, train_idx)
            val_data = torch.utils.data.Subset(train_val_data, val_idx)
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            model = get_trans_model(feature_dim=96, num_types=3, d_model=256, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1).to(device)
            criterion = loglikelihood_trans
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            for epoch in range(1, EPOCH + 1):
                train_model(model, train_loader, criterion, optimizer, 1)
                
                if epoch % 20 == 0 or epoch == EPOCH:
                    val_loss, correct, total = evaluate_model(model, val_loader, criterion)
                    val_accuracy = correct / total
                    
                    new_row = pd.DataFrame({'Epoch': [epoch], 'Fold': [f'Outer {outer_fold}, Inner {inner_fold}'], 'Loss': [val_loss], 'Accuracy': [val_accuracy], 'Seq_Len': [seq_len], 'Test_Index': ['N/A']})
                    df = pd.read_excel(excel_filename)
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_excel(excel_filename, index=False)
                    
                    if val_accuracy > best_inner_accuracy:
                        best_inner_accuracy = val_accuracy
                        best_inner_epoch = epoch
                    
                    print(f"Epoch {epoch}, Outer Fold {outer_fold}, Inner Fold {inner_fold}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Evaluate on the test set with the best model
        model = get_trans_model(feature_dim=96, num_types=3, d_model=256, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Retrain the model on the entire training set using the best epoch
        train_loader = torch.utils.data.DataLoader(train_val_data, batch_size=batch_size, shuffle=True)
        for _ in range(best_inner_epoch):
            train_model(model, train_loader, criterion, optimizer, 1)
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_loss, correct, total = evaluate_model(model, test_loader, criterion)
        test_accuracy = correct / total
        
        filename = f"{subject_results_root}/lr{learning_rate}_epoch{best_inner_epoch}_seq{seq_len}_outer{outer_fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as file:
            file.write(f"Subject: {subject}\n")
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Best Epoch: {best_inner_epoch}\n")
            file.write(f"Learning Rate: {learning_rate}\n")
            file.write(f"Sequence Length: {seq_len}\n")
            file.write(f"Test Loss: {test_loss:.4f}\n")
            file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        
        new_row = pd.DataFrame({'Epoch': [best_inner_epoch], 'Fold': [f'Outer {outer_fold} Test'], 'Loss': [test_loss], 'Accuracy': [test_accuracy], 'Seq_Len': [seq_len], 'Test_Index': [test_idx.tolist()]})
        df = pd.read_excel(excel_filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_filename, index=False)
        print(f"Outer Fold {outer_fold} Test Results saved to {filename}")

seq_length = SEQ_L
while seq_length > 10:
    nested_cv(seq_length)
    seq_length = seq_length // 2
