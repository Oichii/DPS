import os.path
import random
import numpy as np
import torch
import lightning.pytorch as pl


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
