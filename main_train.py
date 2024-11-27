import json
import torch
import torch.backends.cudnn as cudnn
from utils import set_seed, create_dir
import lightning.pytorch as pl
from SkinClassifier import SkinClassifier
from SkinMaskDataModule import SkinDataModule
import pandas as pd
from lightning.pytorch import loggers as pl_loggers

with open('config.json') as config_file:
    paths = json.load(config_file)

save_dir = paths['savePath']
create_dir(save_dir)

cpu = False
resume = paths['resumePath']  # model path to resume training

if cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

cudnn.benchmark = True
seed = 68616

cfg = {
    "batch_size": 32,
    'img_size': 224,
    'in_channels': 3,
    'lr': 0.012,
    'momentum': 0.95,
    'weight_decay': 1e-5,
    'epochs': 200,
    "outputs": 9,
    'net_name': 'convnext_mask_9c_test',
}

if __name__ == '__main__':

    torch.set_float32_matmul_precision('high')
    set_seed(seed)
    model = SkinClassifier(True, out=cfg['outputs'], lr=cfg['lr'],
                           batch_size=cfg['batch_size'],
                           wd=cfg['weight_decay'], momentum=cfg['momentum'])
    df = pd.read_csv(paths['csvPath'])

    for fold in range(5):
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

        data_module = SkinDataModule(
            data_dir=paths['imagesPath'],
            mask_dir=paths['masksPath'],
            train_df=df_train,
            val_df=df_valid,
            image_size=cfg['img_size'],
            batch_size=cfg['batch_size']
        )
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='ckpt_best_{validation_loss:02f}-{epoch:02d}-{validation_accuracy:02f}'
                         f'_{cfg["net_name"]}_'
                         f'_lr={cfg["lr"]}'
                         f'_mom={cfg["momentum"]}'
                         f'_wd={cfg["weight_decay"]}'
                         f'_out={cfg["outputs"]}'
                         f'_bs={cfg["batch_size"]}'
                         f'_imgSize={cfg["img_size"]}'
                         f'_fold={fold}'
                         f'_size={cfg["img_size"]}_{seed}',
                monitor='validation_loss',
                mode='min',
                save_on_train_epoch_end=True,
                save_top_k=2
            ),

            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='ckpt_{validation_loss:02f}-{epoch:02d}-{validation_accuracy:02f}'
                         f'_{cfg["net_name"]}_'
                         f'_lr={cfg["lr"]}'
                         f'_mom={cfg["momentum"]}'
                         f'_wd={cfg["weight_decay"]}'
                         f'_out={cfg["outputs"]}'
                         f'_bs={cfg["batch_size"]}'
                         f'_imgSize={cfg["img_size"]}'
                         f'_fold={fold}'
                         f'_size={cfg["img_size"]}_{seed}',

                save_on_train_epoch_end=True,
                every_n_epochs=1,
                save_last=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ]
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs_multiclass_deep_experiments/",
                                                 version=f'{cfg["net_name"]}'
                                                         f'_lr={cfg["lr"]}'
                                                         f'_mom={cfg["momentum"]}'
                                                         f'_wd={cfg["weight_decay"]}'
                                                         f'_out={cfg["outputs"]}'
                                                         f'_bs={cfg["batch_size"]}'
                                                         f'_imgSize={cfg["img_size"]}'
                                                         f'_fold={fold}'
                                                         f'_size={cfg["img_size"]}_{seed}')

        trainer = pl.Trainer(callbacks=callbacks, accelerator='gpu', devices=1, max_epochs=cfg['epochs'],
                             logger=tb_logger, log_every_n_steps=50, default_root_dir=save_dir)

        # Fit model
        trainer.fit(model=model, datamodule=data_module)

        model.freeze()
        trainer.validate(model=model, datamodule=data_module)
