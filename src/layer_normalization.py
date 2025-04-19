import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape: int = 768, eps=1e-05):
        super(LayerNormalization, self).__init__()
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        else:
            pass
        
if __name__ == "__main__":
    norm = LayerNormalization(
        normalized_shape=256
    )