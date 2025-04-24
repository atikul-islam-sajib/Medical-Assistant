import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append("./src/")

from utils import device_init, load_files

class Tester():
    def __init__(self, dataset: str = "test", device: str = "cuda"):
        self.dataset = dataset
        self.device = device
        
        self.device = device_init(device=device)
        
    def load_dataset(self):
        path = "./data/processed/"
        if self.dataset == "test":
            test_dataloader = os.path.join(path, "test_dataloader.pkl")
            test_dataloader = load_files(filename=test_dataloader)
            
            return test_dataloader
        
        elif self.dataset == "valid":
            val_dataloader = os.path.join(path, "valid_dataloader.pkl")
            val_dataloader = load_files(filename=val_dataloader)
            
            return val_dataloader

        else:
            raise ValueError("Dataset must be either 'test' or 'valid'")
    
    def load_model(self):
        pass
    
    def test(self):
        pass
        