'''
Validation using the first 3-fold stategy on each individual
This strategy is leveraged specifically for the time-encoded models
'''

import torch
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd

from Dataset import get_dataloaders,get_dataloaders_trans,get_trainloader_single,get_valloader_single
from Models import get_CNN_model,get_non_pad_mask,get_trans_model
from Utils import loglikelihood,loglikelihood_trans
from tqdm import tqdm
from datetime import datetime
from Main import train_model,evaluate_model

# 常量定义
from Consts import TRAIN_ROOT,BATCH , EPOCH, LR, NUM_CLASS, SEQ_L, CV1_SUBJECT

# 超参数设置
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cv1(seq_len):
    subject = CV1_SUBJECT
    datasets = ['1', '2', '3']
    root=TRAIN_ROOT
    batch_size = BATCH
    num_epochs = EPOCH
    learning_rate = LR
    num_classes = NUM_CLASS  # 根据实际情况调整类别数
    seq_len=seq_len
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 创建结果目录
    subject_results_root=f"with_temp_enc/{subject}/evaluation_results_{seq_len}"
    path = os.path.dirname(subject_results_root)
    if not os.path.exists(subject_results_root):
        os.makedirs(subject_results_root)
    
    # 创建Excel文件
    excel_filename = f"{subject_results_root}/lr{learning_rate}_seq{seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = os.path.dirname(excel_filename)
    if not os.path.exists(path):
        os.makedirs(path)
    df = pd.DataFrame(columns=['Epoch', 'Loss', 'Accuracy'])
    df.to_excel(excel_filename, index=False)

    for num_epochs in range (EPOCH,0,-50):
        loss_subject = 0
        correct_subject = 0
        total_subject = 0
        subject_accuracy=0
        for i in range(3):
            # Prepare train and validation datasets
            train_datasets = [datasets[j] for j in range(3) if j != i]
            val_dataset = datasets[i]

            # 初始化模型、损失函数和优化器
            #CNNmodel = get_CNN_model(num_classes).to(device)
            Transformer_model = get_trans_model(feature_dim=96, num_types=3, d_model=256, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1).to(device)
            criterion = loglikelihood_trans
            optimizer = optim.Adam(Transformer_model.parameters(), lr=learning_rate)

            # Train on the two datasets
            for train_dataset in train_datasets:
                train_mat=root+ '/' +subject+'/'+ train_dataset + '.mat'
                print("Begin Train on "+subject+" Training Set "+train_dataset)
                train_loader = get_trainloader_single(train_mat, batch_size,seq_len)
                train_model(Transformer_model, train_loader, criterion, optimizer, num_epochs)

            print("eeg_weight:", Transformer_model.encoder.eeg_weight.item())
            print("tem_weight:", Transformer_model.encoder.tem_weight.item())

            # Evaluate on the third dataset
            val_mat= root + '/' +subject+'/'+ val_dataset + '.mat'
            val_loader = get_valloader_single(val_mat, batch_size,seq_len)
            print("Begin Evaluation on "+subject+" Training Set "+val_dataset)
            loss, correct_single,predicted_single= evaluate_model(Transformer_model, val_loader, criterion)
            correct_subject += correct_single
            total_subject += predicted_single
            loss_subject += loss


        filename = f"{subject_results_root}/lr{learning_rate}_epoch{num_epochs}_seq{seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        loss_subject /= 3
        subject_accuracy = correct_subject / total_subject

        with open(filename, 'w') as file:
            file.write(f"Subject: {subject}\n")
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Number of Epochs: {num_epochs}\n")
            file.write(f"Learning Rate: {learning_rate}\n")
            file.write(f"Sequence Length: {seq_len}\n")
            file.write(f"Total Loss: {loss_subject:.4f}\n")
            file.write(f"Total Accuracy: {subject_accuracy:.4f}\n")

        print(f'Results saved to {filename}')
        new_row = pd.DataFrame({'Epoch': [num_epochs], 'Loss': [loss_subject], 'Accuracy': [subject_accuracy]})
        df = pd.read_excel(excel_filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_filename, index=False)
    

seq_length = SEQ_L
while seq_length > 10:
    cv1(seq_length)
    seq_length = seq_length // 2