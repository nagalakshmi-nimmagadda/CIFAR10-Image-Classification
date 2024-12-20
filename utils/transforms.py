import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

class Transforms:
    def __init__(self, means, stds):
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=4,  # BORDER_REFLECT_101
                p=0.7
            ),
            A.CoarseDropout(
                max_holes=2,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=4,
                min_width=4,
                fill_value=tuple([x * 255.0 for x in means]),
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ], p=0.3),
            A.Normalize(mean=means, std=stds),
            ToTensorV2()
        ])

        self.test_transforms = A.Compose([
            A.Normalize(mean=means, std=stds),
            ToTensorV2()
        ])

    def __call__(self, img, train=True):
        img = np.array(img)
        if train:
            transformed = self.train_transforms(image=img)
        else:
            transformed = self.test_transforms(image=img)
        return transformed["image"] 