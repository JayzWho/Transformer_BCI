import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import Dataset

'''Datasets for a simple CNN Model'''
# Define a class for time series dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, mat_file):
        # Load the mat file using scipy.io
        self.data = scio.loadmat(mat_file)
        # Extract features from X variable
        self.features = self.data['X']
        # Extract labels from Y variable (assumed to be a column vector)
        self.labels = self.data['Y']
        self.mapping = {2: 0, 3: 1, 7: 2}
        
        # Apply the mapping to labels
        self.labels = self.map_labels(self.labels)

        self.labels = self.one_hot_encode(self.labels, 3)
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
        return torch.tensor(one_hot)  # Shape: (y,num_classes)

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

def get_trainloader_CNN_single(train_mat,batch_size):
    train_dataset = TimeSeriesDataset(train_mat)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
def get_testloader_CNN_single(test_mat,batch_size):
    test_dataset = TimeSeriesDataset(test_mat)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
def get_CNN_dataset(mat_file):
    dataset=TimeSeriesDataset(mat_file)
    return dataset

'''Datasets for the Transformer Model'''
class TransDataset(Dataset):
    def __init__(self, mat_file, seq_len):
        """
        Args:
            mat_file (str): Path to the .mat file.
            seq_len (int): Sequence length for the Transformer model.
        """
        self.data = scio.loadmat(mat_file)
        self.X = self.data['X']  # Assumes X is the features matrix
        self.Y = self.data['Y']  # Assumes Y is labels matrix
        self.seq_len = seq_len

        assert self.X.shape[0] == self.Y.shape[0], "Mismatch in number of samples between X and Y"

        self.num_samples = self.X.shape[0]

    def __len__(self):
        return (self.num_samples + self.seq_len - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        if end_idx > self.num_samples:
            # Calculate the number of missing elements
            missing_len = end_idx - self.num_samples

            # Pad the features and labels with zeros
            features = np.zeros((self.seq_len, self.X.shape[1]), dtype=self.X.dtype)
            labels = np.zeros((self.seq_len, 1), dtype=self.Y.dtype)
            times = np.zeros(self.seq_len, dtype=int)

            # Fill the available data
            available_len = self.num_samples - start_idx
            features[:available_len] = self.X[start_idx:self.num_samples]
            labels[:available_len] = self.Y[start_idx:self.num_samples]
            times[:available_len] = np.arange(start_idx, self.num_samples)+1
        else:
            features = self.X[start_idx:end_idx, :]
            labels = self.Y[start_idx:end_idx]
            times = np.arange(start_idx, end_idx)+1

        labels = torch.tensor(labels, dtype=torch.long)

        mapping = {2: 0, 3: 1, 7: 2}

        # Map the original labels to new labels
        mapped_labels = labels.clone()
        for original, new in mapping.items():
            mapped_labels[labels == original] = new

        # Remove the last dimension to fit the one_hot function
        mapped_labels = mapped_labels.squeeze(-1)

        # Perform one-hot encoding
        one_hot_labels = torch.nn.functional.one_hot(mapped_labels, num_classes=3)

        return {
            'features': torch.tensor(features, dtype=torch.float32),  # Shape: [seq_len, feature_dim]
            'times': torch.tensor(times, dtype=torch.long),  # Shape: [seq_len]
            'labels': torch.tensor(one_hot_labels, dtype=torch.long)  # Shape: [seq_len]
        }
    
def get_dataloaders_trans(train_mat_file, val_mat_file, batch_size, seq_len):
    # Create datasets
    train_dataset = TransDataset(train_mat_file, seq_len)
    val_dataset = TransDataset(val_mat_file, seq_len)
        
    # Create data loaders
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_trainloader_single(mat_file, batch_size, seq_len):
    dataset = TransDataset(mat_file, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_combined_dataset(mat_file, seq_len):
    dataset=TransDataset(mat_file, seq_len)
    return dataset 

def get_valloader_single(mat_file, batch_size, seq_len):
    dataset = TransDataset(mat_file, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader