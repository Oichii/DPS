import numpy as np
# from datasets.SkinMaskDataset import SkinDataset, get_transforms
from datasets.skinDataset import SkinDataset, get_transforms
import lightning.pytorch as pl
from torch.utils.data import DataLoader, WeightedRandomSampler


class SkinDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, mask_dir: str, train_df='', val_df='', test_df='', train_path='', batch_size=32,
                 image_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms_train, self.transforms_val = get_transforms(image_size)

        class_weights = {}
        for c in np.unique(train_df['target']):
            class_weights[c] = 1 / len(train_df[train_df['target'] == c])

        self.valid_dataset = SkinDataset(val_df,
                                         img_path=self.data_dir,
                                         transforms=self.transforms_val,
                                         # masks_path=mask_dir
                                         )
        self.train_dataset = SkinDataset(train_df,
                                         img_path=self.data_dir,
                                         transforms=self.transforms_train,
                                         # masks_path=mask_dir
                                         )

        if test_df != '':
            self.test_dataset = SkinDataset(test_df,
                                            ext='.jpg',
                                            img_path=train_path,
                                            transforms=self.transforms_val,
                                            test=True
                                            )

        self.weighted_sampler = WeightedRandomSampler(
            weights=[class_weights[i] for i in train_df.target],
            num_samples=len(train_df),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.weighted_sampler, drop_last=True,
                          shuffle=False, num_workers=3, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=3, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=3, persistent_workers=True)
