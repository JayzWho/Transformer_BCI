import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset

# Define a class for time series dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, mat_file):
        # Load the mat file using scipy.io
        self.data = scipy.io.loadmat(mat_file)
        # Extract features from X variable
        self.features = self.data['X']
        # Extract labels from Y variable (assumed to be a column vector)
        self.labels = self.data['Y']
        self.mapping = {2: 0, 3: 1, 7: 2}
        
        # Apply the mapping to labels
        self.labels = self.map_labels(self.labels)

        self.labels = self.one_hot_encode(self.labels, 3)
        self.labels = self.labels.squeeze(1)
        # Generate time indices
        self.time_indices = list(range(self.features.shape[0]))
    
    def map_labels(self, labels):
        # Apply the mapping to each label
        vectorized_mapping = np.vectorize(self.mapping.get)
        return vectorized_mapping(labels.flatten()).reshape(labels.shape)
    
    def one_hot_encode(self, labels, num_classes):
        # Convert labels to one-hot encoding
        num_samples = labels.shape[0]
        one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
        one_hot[np.arange(num_samples), labels.flatten()] = 1
        # Convert to tensor and add the extra dimension
        return torch.tensor(one_hot).unsqueeze(1)  # Shape: (y, 1, num_classes)

    def __len__(self):
        # Return the length of the labels
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert the features to a tensor with float32 data type
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        # Convert the label to a tensor with long data type
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # Return the features, label, and time index
        return features, label, self.time_indices[idx]

# Define a function to get data loaders
def get_dataloaders(train_mat, val_mat, batch_size):
    # Create a train dataset
    train_dataset = TimeSeriesDataset(train_mat)
    # Create a validation dataset
    val_dataset = TimeSeriesDataset(val_mat)
    
    # Create a train data loader with batch size and shuffle
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Create a validation data loader with batch size and no shuffle
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Return the train and validation data loaders
    return train_loader, val_loader
