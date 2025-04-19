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
    parser = argparse.ArgumentParser(description="Feed Forward Neural Network for the Medical Assistant".title())
    parser.add_argument("--d_model", type=int, default=768, help="The dimension of the input features")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="The dimension of the output features")
    parser.add_argument("--dropout", type=float, default=0.1, help="The dropout rate")
    
    args = parser.parse_args()
    
    image_size = 224
    patch_size = 16
    image_channels = 1
    dropout = 0.1
    
    total_patches = (image_size // patch_size) ** 2
    dimension = (image_channels * patch_size ** 2)
    dim_feedforward = 4 * dimension
    
    images = torch.randn(image_channels // image_channels, total_patches, dimension)
    
    network = FeedForwardNeuralNetwork(
        d_model=dimension,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    assert (network(images).size()) == images.size(), "FFNN is not working properly".capitalize()
