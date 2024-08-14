import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import pandas as pd

# 常量定义
from Consts import TRAIN_ROOT,BATCH , EPOCH, LR, NUM_CLASS, SEQ_L, SUBJECT
from Dataset import get_trainloader_single,get_valloader_single,get_combined_dataset,get_testloader_CNN_single,get_trainloader_CNN_single
from Models import get_non_pad_mask,get_trans_model,get_CNN_model
from Utils import loglikelihood_trans,loglikelihood
from Main import train_model

# Hyperparameters (you can modify these based on your best set of hyperparameters)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_seq_len = SEQ_L  # Replace with your best sequence length
best_lr = LR   # Replace with your best learning rate
best_num_epochs = EPOCH  # Replace with your best number of epochs
batch_size = BATCH
num_classes = NUM_CLASS
subject=SUBJECT
root = TRAIN_ROOT
datasets = ['1', '2', '3']
source='saved'
pth_root='model_params_from_cv1'
pth_model='time_encoded'

def evaluate_model(model, val_loader):
    model.eval()
    all_valid_predicted = []
    all_valid_single_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            features = batch_data['features']
            times = batch_data['times']
            labels = batch_data['labels']

            features = features.to(device)
            times = times.to(device)
            labels = labels.to(device)

            mask = get_non_pad_mask(times)
            outputs = model.forward(labels, features, times)
            
            # 计算预测类别
            _, predicted = torch.max(outputs.data, 2)  # 在 num_classes 维度上取最大值的索引，即预测类别

            # 计算真实类别
            single_labels = torch.argmax(labels, dim=2)  # 在 num_classes 维度上取最大值的索引，即真实类别

            # 将 mask 扩展到与 predicted 和 single_labels 形状相同
            mask = mask.squeeze(-1)  # 将 mask 从 [batch, seq_len, 1] 变为 [batch, seq_len]

            # 只保留 mask 中有效的位置
            valid_predicted = predicted[mask == 1]
            valid_single_labels = single_labels[mask == 1]

            # 将每个批次的有效预测结果和真实标签添加到列表中
            all_valid_predicted.append(valid_predicted.cpu().numpy())
            all_valid_single_labels.append(valid_single_labels.cpu().numpy())

    # 将所有批次的预测结果和真实标签拼接在一起
    all_valid_predicted = np.concatenate(all_valid_predicted)
    all_valid_single_labels = np.concatenate(all_valid_single_labels)
    
    return all_valid_single_labels, all_valid_predicted

def test_model():
    test_mat = f"{root}/{subject}/test.mat"  # Update with your test data path
    # Load data
    
    test_loader = get_valloader_single(test_mat, batch_size, best_seq_len)

    # Initialize model with the specified hyperparameters
    model = get_trans_model(feature_dim=96, num_types=num_classes, d_model=256, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1).to(device)
    

    # Define loss function and optimizer
    criterion = loglikelihood_trans  # Replace with your actual loss function
    
    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    if source=='saved':
        pth_path=f'{pth_root}/{pth_model}/{subject}_cv1_model_parameters.pth'
        model.load_state_dict(torch.load(pth_path))
    else:
        # Train the model
        for train_dataset in datasets:
            train_mat=root+ '/' +subject+'/'+ train_dataset + '.mat'
            print("Begin Train on "+subject+" Training Set "+train_dataset)
            train_loader = get_trainloader_single(train_mat, batch_size,best_seq_len)
            train_model(model, train_loader, criterion, optimizer,best_num_epochs)

        # Save the model parameters
        print("Saving model parameters...")
        torch.save(model.state_dict(), f"{subject}_cv1_model_parameters.pth")
        print("Model parameters saved.")














    # Evaluate on the test set
    true_labels, predicted_labels= evaluate_model(model, test_loader)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report
    report = classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1', 'Class 2'], output_dict=True)
    print(classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1', 'Class 2']))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
    print("Confusion Matrix:\n", conf_matrix)

    # Save classification report and confusion matrix to an Excel file
    report_df = pd.DataFrame(report).transpose()

    # Create a DataFrame for the confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Actual {i}' for i in range(len(conf_matrix))],
                                columns=[f'Predicted {i}' for i in range(len(conf_matrix))])

     # Save both classification report and confusion matrix to an Excel file
    excel_filename = f"{subject}_classification_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(excel_filename) as writer:
        report_df.to_excel(writer, sheet_name='Classification Report')
        conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')

    print(f"Classification report and confusion matrix saved to {excel_filename}")


    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[f'Predicted {i}' for i in range(num_classes)],
                yticklabels=[f'Actual {i}' for i in range(num_classes)],
                annot_kws={"size": 18})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plot_filename = f"{subject}_confusion_matrix_acc{accuracy:.4f}.png"
    plt.savefig(plot_filename)
    plt.show()
    
# Run the test
test_model()