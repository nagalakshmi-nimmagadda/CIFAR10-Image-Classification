import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np
from torchinfo import summary

from models.custom_resnet import CustomCNN
from utils.data_loader import CIFAR10DataLoader
from utils.transforms import Transforms
from utils.train import Trainer
from config import CONFIG, CIFAR10_MEAN, CIFAR10_STD

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Initialize transforms
    transforms = Transforms(CIFAR10_MEAN, CIFAR10_STD)
    
    # Get data loaders
    data_loader = CIFAR10DataLoader(
        transform=transforms,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )
    train_loader, test_loader = data_loader.get_loaders()
    
    # Initialize model
    model = CustomCNN().to(CONFIG['device'])
    
    # Print dataset and model information
    print("\n" + "="*50)
    print("Dataset Information:")
    print(f"Training Set Size: {len(train_loader.dataset):,} images")
    print(f"Test Set Size: {len(test_loader.dataset):,} images")
    print(f"Number of Classes: 10")
    print(f"Input Image Size: 32x32x3")
    
    print("\nModel Information:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    print("\nTraining Configuration:")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Number of Epochs: {CONFIG['epochs']}")
    print(f"Learning Rate: {CONFIG['learning_rate']}")
    print(f"Device: {CONFIG['device']}")
    print("="*50 + "\n")
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        epochs=CONFIG['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Warm up for 20% of training
        div_factor=10,  # Initial lr = max_lr/10
        final_div_factor=100  # Min lr = initial_lr/100
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=CONFIG['device'],
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # After model initialization and before training
    print("\nModel Summary:")
    
    summary(model, input_size=(1, 3, 32, 32), 
           col_names=["input_size", "output_size", "num_params", "kernel_size"],
           depth=4,
           device=CONFIG['device'])
    
    # Training loop
    best_acc = 0
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f'\nEpoch: {epoch}')
        train_loss, train_acc = trainer.train_epoch()
        test_loss, test_acc = trainer.test()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main() 