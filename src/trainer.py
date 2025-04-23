import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

from utils import device_init
from helper import helper

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
        
        
        
    def train(self):
        pass
