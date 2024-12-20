import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial block (C1) - Efficient feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # RF: 3x3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DepthwiseSeparableConv(16, 32, 3),  # RF: 5x5
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Second block (C2) - Spatial reduction
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, 3),  # RF: 7x7
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, stride=2, padding=1),  # RF: 11x11
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # Third block (C3) - Dilated convolutions for RF
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(48, 64, 3),  # RF: 15x15
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),  # RF: 23x23
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Final block (C40) - Feature refinement
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(64, 96, 3, stride=2),  # RF: 31x31
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1),  # Point-wise conv
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.05)  # Very light dropout
        )
        self.fc = nn.Linear(96, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 96)
        x = self.fc(x)
        return x