import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, nheads: int = 6, dimension: int = 768):
        super(MultiHeadAttentionLayer, self).__init__()
        self.nheads = nheads
        self.dimension = dimension
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        else:
            pass
        
if __name__ == "__main__":
    multihead_attention = MultiHeadAttentionLayer(
        nheads=8,
        dimension=768
    )