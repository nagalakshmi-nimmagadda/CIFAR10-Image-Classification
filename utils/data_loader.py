import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None, train=True):
        self.dataset = dataset
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image, self.train)
        return image, label

class CIFAR10DataLoader:
    def __init__(self, transform, batch_size=128, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Get datasets
        train_dataset = datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True,
            transform=None
        )
        
        test_dataset = datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True,
            transform=None
        )

        self.train_dataset = CIFAR10Dataset(train_dataset, transform, train=True)
        self.test_dataset = CIFAR10Dataset(test_dataset, transform, train=False)
    
    def get_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        return train_loader, test_loader 