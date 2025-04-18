import os
import sys
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import dataloader

sys.path.append("./src/")


class Loader:
    def __init__(
        self,
        image_path: str = "./data/raw",
        image_channels: int = 3,
        image_size: int = 224,
        batch_size: int = 64,
        split_size: float = 0.25,
    ):
        self.image_path = image_path
        self.image_channels = image_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size
        
    def unzip_folder(self):
        pass
    
    def split_dataset(self, **kwargs):
        pass
    
    def image_transforms(self):
        pass
    
    def create_dataloader(self):
        pass
    
if __name__ == "__main__":
    loader = Loader(
        image_path="./data/raw/dataset.zip",
        image_channels=3,
        image_size=224,
        batch_size=64,
        split_size=0.25,
    )