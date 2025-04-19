import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation_func = activation
        
        if self.activation_func == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_func == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif self.activation_func == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation function".capitalize())
        
    def forward(self, x: torch.Tensor)
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    else:
        ...
        
if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(
        dim_feedforward=2048,
        dropout=0.1,
    )
