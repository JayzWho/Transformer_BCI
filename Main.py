import torch
import torch.optim as optim
import torch.nn as nn
from Dataset import get_dataloaders
from Models import get_CNN_model
from Utils import loglikelihood
from tqdm import tqdm
# 常量定义
from Consts import TRAIN_ROOT

# 超参数设置
train_mat = TRAIN_ROOT + '/1.mat'
val_mat = TRAIN_ROOT + '/2.mat'
batch_size = 16
num_epochs = 200
learning_rate = 1e-4
num_classes = 3  # 根据实际情况调整类别数

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据加载器
train_loader, val_loader = get_dataloaders(train_mat, val_mat, batch_size)

# 初始化模型、损失函数和优化器
CNNmodel = get_CNN_model(num_classes).to(device)
criterion = loglikelihood
optimizer = optim.Adam(CNNmodel.parameters(), lr=learning_rate)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs),desc='Training'):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for features, labels, _ in train_loader:
            features = features.to(device)
            #labels = labels.squeeze(1)  # 变为形状 [batch_size, 3]
            
            #labels=torch.argmax(labels, dim=1)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            _, predicted = torch.max(outputs.data, 1)
            single_labels = torch.argmax(labels, dim=1)
            total_train += labels.size(0)
            correct_train += (predicted == single_labels).sum().item()
        
        accuracy = correct_train / total_train
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

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


# 主程序
def main():
    # 训练模型
    train_model(CNNmodel, train_loader, criterion, optimizer, num_epochs)
    # 评估模型
    evaluate_model(CNNmodel, val_loader, criterion)

if __name__ == '__main__':
    main()
