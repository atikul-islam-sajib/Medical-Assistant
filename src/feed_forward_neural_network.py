import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation_func = activation

        self.in_features = self.d_model
        self.out_features = self.dim_feedforward

        if self.activation_func == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_func == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif self.activation_func == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation function".capitalize())

        self.layers = []

        for index in range(2):
            self.layers += [
                nn.Linear(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    bias=False,
                )
            ]
            
            if index == 0:
                self.layers += [self.activation]
                self.layers += [nn.Dropout(p=self.dropout)]
                
            self.in_features = self.out_features
            self.out_features = self.d_model

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            x = self.network(x)
            return x


if __name__ == "__main__":
    network = FeedForwardNeuralNetwork(
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,
    )
    print(network)
