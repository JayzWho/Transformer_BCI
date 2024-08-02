import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(64 * 22, 128)  # Adjust the size according to your input dimensions
        self.fc2 = nn.Linear(128, self.num_classes)

        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        # Initialize biases to zero
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)


    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension, shape: [batch_size, 1, 96]
        #print(f'After unsqueeze: {x.shape}')
        x = self.pool(F.relu(self.conv1(x)))
        #print(f'After conv1 and pool: {x.shape}')
        x = self.pool(F.relu(self.conv2(x)))
        #print(f'After conv2 and pool: {x.shape}')
        x = x.view(x.size(0), -1)  # Flatten
        #print(f'After flatten: {x.shape}')
        x = F.relu(self.fc1(x))
        #print(f'After fc1: {x.shape}')
        x = self.fc2(x)
        #print(f'After fc2: {x.shape}')
        x = F.softmax(x, dim=1)
        
        return x

def get_CNN_model(num_classes):
    model = SimpleCNN(num_classes)
    return model

