import torch

CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.05,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers': 4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'seed': 42,
    'pin_memory': True,
    'prefetch_factor': 2,
    'persistent_workers': True
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616) 