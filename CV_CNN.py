import torch
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd
from sklearn.model_selection import KFold
from Dataset import get_CNN_dataset
from Models import get_CNN_model
from Utils import loglikelihood
from tqdm import tqdm
from datetime import datetime
from Consts import TRAIN_ROOT, BATCH, EPOCH, LR, NUM_CLASS, SUBJECT

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_CNNmodel(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs),desc='Training'):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for features, labels, times in train_loader:
            features = features.to(device)
            
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model.forward(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #output [batch,3]
            _, predicted = torch.max(outputs.data, 1)
            single_labels = torch.argmax(labels, dim=1)
            total_train += labels.size(0)
            correct_train += (predicted == single_labels).sum().item()

        accuracy = correct_train / total_train
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        
# 评估函数
def evaluate_CNNmodel(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels, _ in val_loader:
            features = features.to(device)
            #labels = labels.squeeze(1)  # 变为形状 [batch_size, 3]
            labels = labels.to(device)

            outputs = model.forward(features)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            
            _, predicted = torch.max(outputs.data, 1)
            single_labels=torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == single_labels).sum().item()

    accuracy = correct / total
    print(f'Validation')
    print(f'Validation Loss: {total_loss/len(val_loader):.4f}, Accuracy: {100 * accuracy:.2f}%')
    return total_loss,correct,total


def nested_cv():
    subject = SUBJECT
    datasets = ['1', '2', '3']
    root = TRAIN_ROOT
    batch_size = BATCH
    learning_rate = LR
    num_classes = NUM_CLASS  # Adjust based on actual classes
   
    
    # Create result directories
    subject_results_root = f"CNN/{subject}/evaluation_results"
    if not os.path.exists(subject_results_root):
        os.makedirs(subject_results_root)
    
    # Create Excel file
    excel_filename = f"{subject_results_root}/lr{learning_rate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if not os.path.exists(os.path.dirname(excel_filename)):
        os.makedirs(os.path.dirname(excel_filename))
    df = pd.DataFrame(columns=['Epoch', 'Fold', 'Loss', 'Accuracy', 'Test_Index'])
    df.to_excel(excel_filename, index=False)
    
    # Combine datasets
    combined_data = []
    for dataset in datasets:
        data_mat = os.path.join(root, subject, f"{dataset}.mat")
        combined_data.append(get_CNN_dataset(data_mat))
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
            
            model = get_CNN_model(3).to(device)
            criterion = loglikelihood
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            for epoch in range(1, EPOCH + 1):
                train_CNNmodel(model, train_loader, criterion, optimizer, 1)
                
                if epoch % 20 == 0 or epoch == EPOCH:
                    val_loss, correct, total = evaluate_CNNmodel(model, val_loader, criterion)
                    val_accuracy = correct / total
                    
                    new_row = pd.DataFrame({'Epoch': [epoch], 'Fold': [f'Outer {outer_fold}, Inner {inner_fold}'], 'Loss': [val_loss], 'Accuracy': [val_accuracy],  'Test_Index': ['N/A']})
                    df = pd.read_excel(excel_filename)
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_excel(excel_filename, index=False)
                    
                    if val_accuracy > best_inner_accuracy:
                        best_inner_accuracy = val_accuracy
                        best_inner_epoch = epoch
                    
                    print(f"Epoch {epoch}, Outer Fold {outer_fold}, Inner Fold {inner_fold}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Evaluate on the test set with the best model
        model = get_CNN_model(3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Retrain the model on the entire training set using the best epoch
        train_loader = torch.utils.data.DataLoader(train_val_data, batch_size=batch_size, shuffle=True)
        train_CNNmodel(model, train_loader, criterion, optimizer, best_inner_epoch)
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_loss, correct, total = evaluate_CNNmodel(model, test_loader, criterion)
        test_accuracy = correct / total
        
        filename = f"{subject_results_root}/lr{learning_rate}_epoch{best_inner_epoch}_outer{outer_fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as file:
            file.write(f"Subject: {subject}\n")
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Best Epoch: {best_inner_epoch}\n")
            file.write(f"Learning Rate: {learning_rate}\n")
            file.write(f"Test Loss: {test_loss:.4f}\n")
            file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        
        new_row = pd.DataFrame({'Epoch': [best_inner_epoch], 'Fold': [f'Outer {outer_fold} Test'], 'Loss': [test_loss], 'Accuracy': [test_accuracy], 'Test_Index': [test_idx.tolist()]})
        df = pd.read_excel(excel_filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_filename, index=False)
        print(f"Outer Fold {outer_fold} Test Results saved to {filename}")

nested_cv()