import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

from utils import device_init
from helper import helper, Criterion
from ViT import ViTWithClassifier


class Trainer:
    def __init__(
        self,
        model=None,
        lr: float = 0.001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        momentum: float = 0.85,
        adam: bool = True,
        SGD: bool = False,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.verbose = verbose

        self.init = helper(
            model=self.model,
            lr=self.lr,
            adam=self.adam,
            beta1=self.beta1,
            beta2=self.beta2,
            weight_decay=self.weight_decay,
            SGD=self.SGD,
            momentum=self.momentum,
        )
        
        self.device = device_init(device=device)
        
        self.train_dataloader = self.init["dataloader"]["train_dataloader"]
        self.test_dataloader = self.init["dataloader"]["test_dataloader"]
        self.valid_dataloader = self.init["dataloader"]["valid_dataloader"]
        
        self.optimizer = self.init["optimizer"]
        self.criterion = self.init["criterion"]
        
        self.train_dataloader = self.train_dataloader.to(self.device)
        self.test_dataloader = self.test_dataloader.to(self.device)
        self.valid_dataloader = self.valid_dataloader.to(self.device)

        self.classifier = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        assert self.train_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
        assert self.test_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
        assert self.valid_dataloader.__class__ == torch.utils.data.dataloader.DataLoader

        assert self.classifier.__class__ == ViTWithClassifier
        assert self.criterion.__class__ == Criterion
        
        if self.adam:
            assert self.optimizer.__class__ == torch.optim.Adam
        elif self.SGD:
            assert self.optimizer.__class__ == torch.optim.SGD
        else:
            raise ValueError("Optimizer not supported".capitalize())
        
    def train(self):
        pass
