import torch
from torch import nn
from torchvision import models


class SkinConvNext(nn.Module):
    def __init__(self, pretrained=True, map_size=14, num_classes=2):
        super(SkinConvNext, self).__init__()
        convNext = models.convnext_tiny(pretrained=pretrained)
        features = [*convNext.features.children()]
        self.enc = nn.Sequential(*features[:6])
        self.dec = nn.Conv2d(384, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.classifier = nn.Linear(map_size*map_size, num_classes)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        dec_max = torch.max(dec, dim=1)[0].unsqueeze(dim=1)
        out = self.classifier(torch.flatten(dec_max, start_dim=1))

        return out, dec