import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

class ViT(nn.Module):
    def __init__(
        self,
        encoder_layer: int = 4,
        nhead: int = 8,
        d_model: int = 768,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(ViT, self).__init__()
        self.encoder_layer = encoder_layer
        self.nhead = nhead
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            ...
            
if __name__ == "__main__":
    vit = ViT(
        encoder_layer=4,
        nhead=8,
        d_model=768,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
        bias=False,
        
    )