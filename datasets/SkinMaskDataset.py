from torch.utils.data import Dataset
import os
import cv2
import albumentations as albu
import torch


class SkinDataset(Dataset):
    def __init__(self, dataframe, root_dir='', img_path='', masks_path='', classification=True,
                 transforms=None, additional_channels=False, ext='.jpg', test=False):
        self.df = dataframe

        self.img_path = os.path.join(root_dir, img_path)
        self.mask_path = masks_path
        self.ext = ext
        self.test = test

        self.classification = classification
        self.additional_channels = additional_channels

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = albu.Compose([
                albu.Normalize()
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        image = cv2.imread(os.path.join(self.img_path, self.df.iloc[item, 0] + self.ext))
        mask = cv2.imread(os.path.join(self.mask_path, self.df.iloc[item, 0] + '_mask.png'), 0)
        label = self.df.iloc[item, 4]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)

        mask = cv2.resize(mask, (14, 14))
        mask = torch.tensor(mask) / 255.0
        mask = torch.round(mask)

        label = torch.tensor(label, dtype=torch.long)
        mask = mask * label
        mask = mask.type(torch.long)

        return image, mask, label, self.df.iloc[item, 0]

    def get_labels(self):
        return self.df.iloc[:, 4]


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
    ],
        additional_targets={"mask": "mask"}
    )

    transforms_val = albu.Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize()
    ],
        additional_targets={"mask": "mask"}
    )

    return transforms_train, transforms_val

