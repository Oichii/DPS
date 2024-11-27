from torch.utils.data import Dataset
import os
import cv2
import albumentations as albu
import torch
import numpy as np


class SkinDataset(Dataset):
    def __init__(self, dataframe, root_dir='', img_path='', classification=True,
                 transforms=None, additional_channels=False, color_space='bgr', enhance=False, ext='.jpg', test=False):
        self.df = dataframe

        self.img_path = os.path.join(root_dir, img_path)

        self.classification = classification
        self.additional_channels = additional_channels
        self.colorSpace = color_space
        self.enhance = enhance
        self.ext = ext
        self.test = test

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = albu.Compose([
                albu.Normalize()
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        image = cv2.imread(os.path.join(self.img_path, self.df.iloc[item, 0]+self.ext))
        label = self.df.iloc[item, 4]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)
        image = image['image'].astype(np.float32)
        image = image.transpose(2, 0, 1)

        image = torch.tensor(image).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label, self.df.iloc[item, 1]


def get_transforms(image_size):

    transforms_train = albu.Compose([
        albu.Transpose(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        albu.OneOf([
            albu.MotionBlur(blur_limit=5),
            albu.MedianBlur(blur_limit=5),
            albu.GaussianBlur(blur_limit=(3, 5), sigma_limit=1),
            albu.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albu.OneOf([
            albu.OpticalDistortion(distort_limit=1.0),
            albu.GridDistortion(num_steps=5, distort_limit=1.),
        ], p=0.7),

        albu.CLAHE(clip_limit=4.0, p=0.7),
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albu.Resize(image_size, image_size),
        albu.Normalize()
    ])

    transforms_val = albu.Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize()
    ])

    return transforms_train, transforms_val

