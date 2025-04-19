import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from layer_normalization import LayerNormalization
from multi_head_attention_layer import MultiHeadAttentionLayer
from feed_forward_neural_network import FeedForwardNeuralNetwork


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.nheads = nhead
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
    transformer = TransformerEncoderBlock(
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
        bias=False,
    )