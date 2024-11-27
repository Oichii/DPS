import torch
from torch import optim
import lightning.pytorch as pl
from torchmetrics import Accuracy

from models.SkinConvNext import SkinConvNext
# from models.SkinDenseNet import SkinDenseNet
from losses.DeepPixelWiseLoss import DeepPixelWiseLoss


class SkinClassifier(pl.LightningModule):
    def __init__(self, pretrained=True, map_size=14, out=9, lr=0.03, batch_size=32, wd=0.01, momentum=0.1):
        super().__init__()
        self.lr = lr
        self.map_size = map_size
        self.batch_size = batch_size
        self.wd = wd
        self.momentum = momentum

        self.model = SkinConvNext(pretrained=pretrained, num_classes=out)
        # self.model = SkinDenseNet(pretrained=pretrained, num_classes=out)

        self.criterion = DeepPixelWiseLoss()
        # self.criterion = nn.CrossEntropyLoss()

        self.validation_acc = Accuracy(task="multiclass", num_classes=out)
        self.validation_acc_mean = Accuracy(task="multiclass", num_classes=out)
        self.train_acc = Accuracy(task="multiclass", num_classes=out)

        self.validation_predictions = []
        self.validation_gt = []
        print(self.model)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x_map_true = torch.ones([y.size()[0], self.map_size, self.map_size], device='cuda') * y[:, None, None]
        x_map_true = x_map_true.type(torch.long)

        x_hat, x_map_hat = self.model(x)

        loss = self.criterion(x_hat.squeeze(), y, x_map_hat, x_map_true)

        self.log("train_loss", loss.item(), batch_size=self.batch_size)
        self.train_acc(x_hat.squeeze(), y)
        self.log('train_accuracy', self.train_acc, on_step=True,  batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        x_map_true = torch.ones([y.size()[0], self.map_size, self.map_size], device='cuda') * y[:, None, None]
        x_map_true = x_map_true.type(torch.long)

        x_hat, x_map_hat = self.model(x)

        loss = self.criterion(x_hat, y, x_map_hat, x_map_true)

        self.validation_acc(x_hat, y)
        self.log('validation_accuracy', self.validation_acc, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log("validation_loss", loss.item(), on_epoch=True, on_step=True, prog_bar=True, batch_size=self.batch_size)
        self.validation_predictions.append(x_hat)
        self.validation_gt.append(y)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.lr, weight_decay=self.wd, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        y = self.model(x)
        return y, self.selected_out
