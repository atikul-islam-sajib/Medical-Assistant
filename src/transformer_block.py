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
        d_model: int = 768,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.nheads = nhead
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        self.multi_head_attention = MultiHeadAttentionLayer(
            nhead=self.nheads,
            dimension=self.d_model,
        )

        self.layer_norm1 = LayerNormalization(
            normalized_shape=self.d_model, eps=self.layer_norm_eps
        )
        self.layer_norm2 = LayerNormalization(
            normalized_shape=self.d_model, eps=self.layer_norm_eps
        )
        self.feed_forward_network = FeedForwardNeuralNetwork(
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            residual = x
            
            x = self.multi_head_attention(x)
            x = self.dropout1(x)
            x = torch.add(x, residual)
            x = self.layer_norm1(x)
            
            residual = x
            
            x = self.feed_forward_network(x)
            x = torch.add(x, residual)
            x = self.layer_norm2(x)

            return x

if __name__ == "__main__":
    transformer = TransformerEncoderBlock(
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
        bias=False,
    )
