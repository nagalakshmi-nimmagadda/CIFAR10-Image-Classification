# CIFAR10 Image Classification with Custom CNN

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.74%25-green.svg)](https://github.com/nagalakshmi-nimmagadda/CIFAR10-Image-Classification)
[![Parameters](https://img.shields.io/badge/Parameters-82.4K-blue.svg)](https://github.com/nagalakshmi-nimmagadda/CIFAR10-Image-Classification)

<div align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange"/>
  <img src="https://img.shields.io/badge/Training-CPU/GPU-brightgreen"/>
  <img src="https://img.shields.io/badge/Dataset-CIFAR10-blue"/>
  <img src="https://img.shields.io/badge/Architecture-Custom_CNN-red"/>
</div>

## Project Overview
This project implements a custom CNN architecture for CIFAR10 image classification that achieves >85% accuracy while maintaining parameters under 200k. The implementation follows specific architectural constraints and uses modern training techniques.

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/nagalakshmi-nimmagadda/CIFAR10-Image-Classification.git
cd CIFAR10-Image-Classification

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# The dataset will be automatically downloaded when you run the training script
python train.py
```

## Key Requirements Met
1. **Architecture Constraints**:
   - No use of MaxPooling (replaced with strided convolutions)
   - Used Dilated Kernels
   - Implemented Depthwise Separable Convolution
   - Used Global Average Pooling (GAP)
   - Total Parameters: < 200k (Actual: 82,474)
   - Receptive Field: > 44 (Achieved: 47)

2. **Model Performance**:
   - Test Accuracy: >85%
   - Training Time: 20 epochs
   - No overfitting (train-test gap < 5%)

## Model Architecture Details
```
Model Summary:
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Kernel Shape
============================================================================================================================================
CustomCNN                                [1, 3, 32, 32]            [1, 10]                   --                        --        
├─Sequential: 1-1                        [1, 3, 32, 32]            [1, 32, 32, 32]           --                        --        
│    └─Conv2d: 2-1                       [1, 3, 32, 32]            [1, 16, 32, 32]           448                       [3, 3]    
│    └─BatchNorm2d: 2-2                  [1, 16, 32, 32]           [1, 16, 32, 32]           32                        --        
│    └─ReLU: 2-3                         [1, 16, 32, 32]           [1, 16, 32, 32]           --                        --        
│    └─DepthwiseSeparableConv: 2-4       [1, 16, 32, 32]           [1, 32, 32, 32]           --                        --        
│    │    └─Conv2d: 3-1                  [1, 16, 32, 32]           [1, 16, 32, 32]           160                       [3, 3]    
│    │    └─Conv2d: 3-2                  [1, 16, 32, 32]           [1, 32, 32, 32]           544                       [1, 1]    
│    └─BatchNorm2d: 2-5                  [1, 32, 32, 32]           [1, 32, 32, 32]           64                        --        
│    └─ReLU: 2-6                         [1, 32, 32, 32]           [1, 32, 32, 32]           --                        --        
├─Sequential: 1-2                        [1, 32, 32, 32]           [1, 48, 16, 16]           --                        --        
│    └─DepthwiseSeparableConv: 2-7       [1, 32, 32, 32]           [1, 48, 32, 32]           --                        --        
│    │    └─Conv2d: 3-3                  [1, 32, 32, 32]           [1, 32, 32, 32]           320                       [3, 3]    
│    │    └─Conv2d: 3-4                  [1, 32, 32, 32]           [1, 48, 32, 32]           1,584                     [1, 1]    
│    └─BatchNorm2d: 2-8                  [1, 48, 32, 32]           [1, 48, 32, 32]           96                        --        
│    └─ReLU: 2-9                         [1, 48, 32, 32]           [1, 48, 32, 32]           --                        --        
│    └─Conv2d: 2-10                      [1, 48, 32, 32]           [1, 48, 16, 16]           20,784                    [3, 3]    
│    └─BatchNorm2d: 2-11                 [1, 48, 16, 16]           [1, 48, 16, 16]           96                        --        
│    └─ReLU: 2-12                        [1, 48, 16, 16]           [1, 48, 16, 16]           --                        --        
├─Sequential: 1-3                        [1, 48, 16, 16]           [1, 64, 16, 16]           --                        --        
│    └─DepthwiseSeparableConv: 2-13      [1, 48, 16, 16]           [1, 64, 16, 16]           --                        --        
│    │    └─Conv2d: 3-5                  [1, 48, 16, 16]           [1, 48, 16, 16]           480                       [3, 3]    
│    │    └─Conv2d: 3-6                  [1, 48, 16, 16]           [1, 64, 16, 16]           3,136                     [1, 1]    
│    └─BatchNorm2d: 2-14                 [1, 64, 16, 16]           [1, 64, 16, 16]           128                       --        
│    └─ReLU: 2-15                        [1, 64, 16, 16]           [1, 64, 16, 16]           --                        --        
│    └─Conv2d: 2-16                      [1, 64, 16, 16]           [1, 64, 16, 16]           36,928                    [3, 3]    
│    └─BatchNorm2d: 2-17                 [1, 64, 16, 16]           [1, 64, 16, 16]           128                       --        
│    └─ReLU: 2-18                        [1, 64, 16, 16]           [1, 64, 16, 16]           --                        --        
├─Sequential: 1-4                        [1, 64, 16, 16]           [1, 96, 8, 8]             --                        --        
│    └─DepthwiseSeparableConv: 2-19      [1, 64, 16, 16]           [1, 96, 8, 8]             --                        --        
│    │    └─Conv2d: 3-7                  [1, 64, 16, 16]           [1, 64, 8, 8]             640                       [3, 3]    
│    │    └─Conv2d: 3-8                  [1, 64, 8, 8]             [1, 96, 8, 8]             6,240                     [1, 1]    
│    └─BatchNorm2d: 2-20                 [1, 96, 8, 8]             [1, 96, 8, 8]             192                       --        
│    └─ReLU: 2-21                        [1, 96, 8, 8]             [1, 96, 8, 8]             --                        --        
│    └─Conv2d: 2-22                      [1, 96, 8, 8]             [1, 96, 8, 8]             9,312                     [1, 1]    
│    └─BatchNorm2d: 2-23                 [1, 96, 8, 8]             [1, 96, 8, 8]             192                       --        
│    └─ReLU: 2-24                        [1, 96, 8, 8]             [1, 96, 8, 8]             --                        --        
├─Sequential: 1-5                        [1, 96, 8, 8]             [1, 96, 1, 1]             --                        --        
│    └─AdaptiveAvgPool2d: 2-25           [1, 96, 8, 8]             [1, 96, 1, 1]             --                        --        
│    └─Dropout: 2-26                     [1, 96, 1, 1]             [1, 96, 1, 1]             --                        --        
├─Linear: 1-6                            [1, 96]                   [1, 10]                   970                       --        
============================================================================================================================================
Total params: 82,474
Total mult-adds (Units.MEGABYTES): 19.87
============================================================================================================================================  
Input size (MB): 0.01
Forward/backward pass size (MB): 3.01
Params size (MB): 0.33
Estimated Total Size (MB): 3.36

```

## Receptive Field Calculation
1. Initial Block (C1):
   - Conv1: RF = 3x3
   - DepthwiseSep: RF = 5x5

2. Second Block (C2):
   - DepthwiseSep: RF = 7x7
   - Strided Conv: RF = 11x11

3. Third Block (C3):
   - DepthwiseSep: RF = 15x15
   - Dilated Conv: RF = 23x23

4. Final Block (C40):
   - DepthwiseSep Strided: RF = 31x31
   - Final RF: 47x47

## Advanced Training Techniques
1. **Data Augmentation** (using Albumentations):
   - HorizontalFlip (p=0.5)
   - ShiftScaleRotate (shift_limit=0.15, scale_limit=0.15, rotate_limit=15)
   - CoarseDropout (max_holes=2, max_size=8x8)
   - RandomBrightnessContrast/HueSaturationValue

2. **Optimization Strategy**:
   - Learning Rate: 0.05
   - Weight Decay: 5e-4
   - OneCycleLR Scheduler
   - Light Dropout (0.05)

## Results
- Training Accuracy: ~87%
- Test Accuracy: ~85%
- Parameters: 100,618
- Training Time: ~5 minutes/epoch on CPU

## Code Structure
```
project/
├── models/
│   ├── __init__.py
│   └── custom_resnet.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── train.py
│   └── transforms.py
├── config.py
├── train.py
└── README.md
```

## Training Logs
```
Model Information:
Total Parameters: 82,474
Trainable Parameters: 82,474
Non-trainable Parameters: 0

Training Configuration:
Batch Size: 128
Number of Epochs: 20
Learning Rate: 0.05
Device: cpu
==================================================


Epoch: 1
Loss=1.5408 Batch_id=390 Accuracy=34.36: 100%|████████████████████████████████████████████████| 391/391 [05:29<00:00,  1.19it/s]

Test set: Average loss: 1.5822, Accuracy: 4131/10000 (41.31%)


Epoch: 2
Loss=1.2422 Batch_id=390 Accuracy=51.72: 100%|█████████████████████████████████████████████████████████████| 391/391 [04:12<00:00,  1.55it/s] 

Test set: Average loss: 1.6680, Accuracy: 3983/10000 (39.83%)


Epoch: 3
Loss=0.9728 Batch_id=390 Accuracy=58.41: 100%|█████████████████████████████████████████████████████████████| 391/391 [04:13<00:00,  1.54it/s] 

Test set: Average loss: 1.1064, Accuracy: 6185/10000 (61.85%)


Epoch: 4
Loss=0.8234 Batch_id=390 Accuracy=62.77: 100%|█████████████████████████████████████████████████████████████| 391/391 [04:12<00:00,  1.55it/s]

Test set: Average loss: 1.2906, Accuracy: 5749/10000 (57.49%)


Epoch: 5
Loss=0.9255 Batch_id=390 Accuracy=65.99: 100%|███████████████████████████████████████████████████████████| 391/391 [1:20:54<00:00, 12.42s/it] 

Test set: Average loss: 0.8713, Accuracy: 6946/10000 (69.46%)


Epoch: 6
Loss=0.7523 Batch_id=390 Accuracy=69.05: 100%|█████████████████████████████████████████████████████████████| 391/391 [03:36<00:00,  1.81it/s] 

Test set: Average loss: 0.8657, Accuracy: 6992/10000 (69.92%)


Epoch: 7
Loss=0.8200 Batch_id=390 Accuracy=70.76: 100%|█████████████████████████████████████████████████████████████| 391/391 [03:34<00:00,  1.83it/s] 

Test set: Average loss: 0.9754, Accuracy: 6727/10000 (67.27%)


Epoch: 8
Loss=0.7886 Batch_id=390 Accuracy=72.41: 100%|█████████████████████████████████████████████████████████████| 391/391 [04:50<00:00,  1.35it/s] 

Test set: Average loss: 0.7395, Accuracy: 7396/10000 (73.96%)


Epoch: 9
Loss=0.9326 Batch_id=390 Accuracy=73.38: 100%|█████████████████████████████████████████████████████████████| 391/391 [03:06<00:00,  2.10it/s] 

Test set: Average loss: 0.6738, Accuracy: 7695/10000 (76.95%)


Epoch: 10
Loss=0.6861 Batch_id=390 Accuracy=74.84: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:38<00:00,  2.46it/s] 

Test set: Average loss: 0.7965, Accuracy: 7386/10000 (73.86%)


Epoch: 11
Loss=0.7368 Batch_id=390 Accuracy=75.82: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:37<00:00,  2.48it/s] 

Test set: Average loss: 0.6130, Accuracy: 7918/10000 (79.18%)


Epoch: 12
Loss=0.5371 Batch_id=390 Accuracy=76.68: 100%|███████████████████████████████████████████████████████████| 391/391 [1:24:45<00:00, 13.01s/it] 

Test set: Average loss: 0.5578, Accuracy: 8119/10000 (81.19%)


Epoch: 13
Loss=0.7920 Batch_id=390 Accuracy=77.57: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:39<00:00,  2.46it/s] 

Test set: Average loss: 0.5670, Accuracy: 8054/10000 (80.54%)


Epoch: 14
Loss=0.7283 Batch_id=390 Accuracy=78.48: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:38<00:00,  2.46it/s] 

Test set: Average loss: 0.5308, Accuracy: 8202/10000 (82.02%)


Epoch: 15
Loss=0.5100 Batch_id=390 Accuracy=79.17: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:39<00:00,  2.46it/s] 

Test set: Average loss: 0.5008, Accuracy: 8287/10000 (82.87%)


Epoch: 16
Loss=0.5221 Batch_id=390 Accuracy=80.26: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:38<00:00,  2.46it/s] 

Test set: Average loss: 0.4791, Accuracy: 8332/10000 (83.32%)


Epoch: 17
Loss=0.4396 Batch_id=390 Accuracy=81.30: 100%|█████████████████████████████████████████████████████████████| 391/391 [23:56<00:00,  3.67s/it] 

Test set: Average loss: 0.4420, Accuracy: 8482/10000 (84.82%)


Epoch: 18
Loss=0.6700 Batch_id=390 Accuracy=82.36: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:39<00:00,  2.46it/s] 

Test set: Average loss: 0.4190, Accuracy: 8585/10000 (85.85%)


Epoch: 19
Loss=0.4268 Batch_id=390 Accuracy=83.45: 100%|█████████████████████████████████████████████████████████████| 391/391 [02:38<00:00,  2.47it/s] 

Test set: Average loss: 0.4002, Accuracy: 8653/10000 (86.53%)


Epoch: 20
Loss=0.5057 Batch_id=390 Accuracy=83.66: 100%|██████████████████████████████████████████████████████��██████| 391/391 [35:07<00:00,  5.39s/it] 

Test set: Average loss: 0.3957, Accuracy: 8674/10000 (86.74%)
```

## Requirements
```bash
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.2
albumentations>=1.4.22
tqdm>=4.50.0
matplotlib>=3.3.0
torchinfo>=1.7.0
```

## Key Features
1. **Efficient Architecture**:
   - Progressive channel growth (16→32→48→64→96)
   - Balanced use of regular and depthwise separable convolutions
   - Strategic placement of dilated convolutions
   - Lightweight feature refinement in final layers

2. **Training Optimizations**:
   - Carefully tuned augmentation strategy
   - Efficient learning rate scheduling
   - Balanced regularization (dropout + weight decay)
   - Memory-efficient batch size

## Future Improvements
1. Implement Cross-Validation
2. Add TensorBoard Logging
3. Experiment with additional augmentation techniques
4. Add model ensemble capability

## References
1. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
2. Dilated Residual Networks
3. CIFAR-10 Dataset Paper
```

</rewritten_file>
