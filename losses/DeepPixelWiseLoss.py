from torch import nn


class DeepPixelWiseLoss(nn.Module):
    def __init__(self, lam=0.5):
        super(DeepPixelWiseLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.lam = lam

    def forward(self, x, x_true, x_map, x_map_true):
        pixel_wise = self.criterion(x_map, x_map_true)
        binary = self.criterion(x, x_true)

        return pixel_wise*self.lam + binary*(1-self.lam)
