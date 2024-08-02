import torch
from Dataset import get_dataloaders
from Models import get_CNN_model

# 常量定义
from Consts import TRAIN_ROOT

# 超参数设置
train_mat = TRAIN_ROOT + '/1.mat'
val_mat = TRAIN_ROOT + '/2.mat'
batch_size = 16

# 获取数据加载器
train_loader, val_loader = get_dataloaders(train_mat, val_mat, batch_size)

# 测试数据加载
def test_data_loader(data_loader):
    for features, labels, time_idx in train_loader:
        print(f"Features: {features.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Time Index: {time_idx}")
        break

# 测试训练集
print("Testing training data loader:")
test_data_loader(train_loader)

# 测试验证集
print("\nTesting validation data loader:")
test_data_loader(val_loader)

model = get_CNN_model(num_classes=3)
features = torch.randn(16, 96)  # Example input
outputs = model.forward(features)
print(f'Outputs shape: {outputs.shape}')  # Should be [16, 3]