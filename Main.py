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

# 常量定义
from Consts import TRAIN_ROOT,BATCH , EPOCH, LR, NUM_CLASS, SEQ_L

# 超参数设置
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    
    model.train()
    for epoch in tqdm(range(num_epochs),desc='Training'):
        running_loss = 0.0
        correct_valid = 0
        total_valid = 0
        for batch_idx, batch_data in enumerate(train_loader):
            features=batch_data['features']
            times=batch_data['times']
            labels=batch_data['labels']
            #print(type(features))
            #print(features)
            #print(type(times))
            #print(type(labels))
            features = features.to(device)
            times = times.to(device)
            #labels = labels.squeeze(1)  # 变为形状 [batch_size, 3]
            
            #labels=torch.argmax(labels, dim=1)
            labels = labels.to(device)
            mask=get_non_pad_mask(times)
            optimizer.zero_grad()
            outputs = model.forward(labels,features,times)

            loss = criterion(outputs, labels,mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            '''
            _, predicted = torch.max(outputs.data, 1)
            single_labels = torch.argmax(labels, dim=1)
            total_train += labels.size(0)
            correct_train += (predicted == single_labels).sum().item()
            '''
            # 计算预测类别
            _, predicted = torch.max(outputs.data, 2)  # 在num_classes维度上取最大值的索引，即预测类别

            # 计算真实类别
            single_labels = torch.argmax(labels, dim=2)  # 在num_classes维度上取最大值的索引，即真实类别

            # 将 mask 扩展到与 predicted 和 single_labels 形状相同
            mask = mask.squeeze(-1)  # 将 mask 从 [batch, seq_len, 1] 变为 [batch, seq_len]

            # 只保留 mask 中有效的位置
            valid_predicted = predicted[mask == 1]
            valid_single_labels = single_labels[mask == 1]

            # 计算总的有效预测数
            total_valid += valid_single_labels.numel()  # 有效位置的总数

            # 计算正确的有效预测数
            correct_valid += (valid_predicted == valid_single_labels).sum().item()

            # 计算准确率


        accuracy = correct_valid / total_valid
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_valid = 0
    total_valid = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            features=batch_data['features']
            times=batch_data['times']
            labels=batch_data['labels']
            features = features.to(device)
            times = times.to(device)
            labels = labels.to(device)
            mask=get_non_pad_mask(times)
            outputs = model.forward(labels, features, times)
            loss = criterion(outputs, labels,mask)
            running_loss += loss.item()
            
            # 计算预测类别
            _, predicted = torch.max(outputs.data, 2)  # 在num_classes维度上取最大值的索引，即预测类别

            # 计算真实类别
            single_labels = torch.argmax(labels, dim=2)  # 在num_classes维度上取最大值的索引，即真实类别

            # 将 mask 扩展到与 predicted 和 single_labels 形状相同
            mask = mask.squeeze(-1)  # 将 mask 从 [batch, seq_len, 1] 变为 [batch, seq_len]

            # 只保留 mask 中有效的位置
            valid_predicted = predicted[mask == 1]
            valid_single_labels = single_labels[mask == 1]

            # 计算总的有效预测数
            total_valid += valid_single_labels.numel()  # 有效位置的总数

            # 计算正确的有效预测数
            correct_valid += (valid_predicted == valid_single_labels).sum().item()
    
    accuracy = correct_valid / total_valid
    loss = running_loss / len(val_loader)
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')
    
    return loss, correct_valid, total_valid
'''
# 评估函数
def evaluate_model(model, val_loader, criterion):
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
'''

'''
# 主程序
def main():
    
    # 训练模型
    #train_model(CNNmodel, train_loader, criterion, optimizer, num_epochs)
    # 评估模型
    #evaluate_model(CNNmodel, val_loader, criterion)
    
    train_model(Transformer_model, train_loader, criterion, optimizer, num_epochs)
    loss, accuracy = evaluate_model(Transformer_model, val_loader, criterion)

    # Create a directory to save results if it doesn't exist
    if not os.path.exists(f'evaluation_results_{seq_len}'):
        os.makedirs(f'evaluation_results_{seq_len}')
    
    # Create the filename including hyperparameters
    filename = f"evaluation_results_{seq_len}/batch{batch_size}_epochs{num_epochs}_lr{learning_rate}_seq{seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Write the results to the file
    with open(filename, 'w') as file:
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Number of Epochs: {num_epochs}\n")
        file.write(f"Learning Rate: {learning_rate}\n")
        file.write(f"Sequence Length: {seq_len}\n")
        file.write(f"Validation Loss: {loss:.4f}\n")
        file.write(f"Validation Accuracy: {accuracy:.4f}\n")

    print(f'Results saved to {filename}')
'''

def main():
    subjects = ['subject1', 'subject2', 'subject3']
    datasets = ['1', '2', '3']
    root=TRAIN_ROOT
    batch_size = BATCH
    num_epochs = EPOCH
    learning_rate = LR
    num_classes = NUM_CLASS  # 根据实际情况调整类别数
    seq_len=SEQ_L

    excel_filename = f"without_temp_enc/evaluation_results_{seq_len}/lr{learning_rate}_seq{seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df = pd.DataFrame(columns=['Epoch', 'Loss', 'Accuracy'])
    df.to_excel(excel_filename, index=False)

    for num_epochs in range (EPOCH,0,-10):
        total_loss = 0
        total_num_predicted=0
        toatl_num_correct=0

        
        for subject in subjects:
            loss_subject = 0
            correct_subject = 0
            total_subject = 0
            for i in range(3):
                # Prepare train and validation datasets
                train_datasets = [datasets[j] for j in range(3) if j != i]
                val_dataset = datasets[i]

                # 检查CUDA是否可用
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

                # Evaluate on the third dataset
                val_mat= root + '/' +subject+'/'+ val_dataset + '.mat'
                val_loader = get_valloader_single(val_mat, batch_size,seq_len)
                print("Begin Evaluation on "+subject+" Training Set "+val_dataset)
                loss, correct_single,predicted_single= evaluate_model(Transformer_model, val_loader, criterion)
                correct_subject += correct_single
                total_subject += predicted_single
                loss_subject += loss

            total_loss += loss_subject
            toatl_num_correct+=correct_subject
            total_num_predicted+=total_subject
            total_accuracy=toatl_num_correct/total_num_predicted

        # Save the results
        if not os.path.exists(f'without_temp_enc/evaluation_results_{seq_len}'):
            os.makedirs(f'without_temp_enc/evaluation_results_{seq_len}')

        filename = f"without_temp_enc/evaluation_results_{seq_len}/lr{learning_rate}_epoch{num_epochs}_seq{seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, 'w') as file:
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Number of Epochs: {num_epochs}\n")
            file.write(f"Learning Rate: {learning_rate}\n")
            file.write(f"Sequence Length: {seq_len}\n")
            file.write(f"Total Loss: {total_loss:.4f}\n")
            file.write(f"Total Accuracy: {total_accuracy:.4f}\n")

        print(f'Results saved to {filename}')
        new_row = pd.DataFrame({'Epoch': [num_epochs], 'Loss': [total_loss], 'Accuracy': [total_accuracy]})
        df = pd.read_excel(excel_filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_filename, index=False)
    
if __name__ == '__main__':
    main()
