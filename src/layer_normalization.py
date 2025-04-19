import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape: int = 768, eps=1e-05):
        super(LayerNormalization, self).__init__()

        self.dimension = normalized_shape
        self.eps = eps
        self.alpha = nn.Parameter(
            data=torch.ones(
                (
                    self.dimension // self.dimension,
                    self.dimension // self.dimension,
                    self.dimension,
                )
            ),
            requires_grad=True,
        )
        self.beta = nn.Parameter(
            data=torch.zeros(
                (
                    self.dimension // self.dimension,
                    self.dimension // self.dimension,
                    self.dimension,
                )
            ),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        else:
            mean = torch.mean(x, dim=-1)
            variance = torch.var(x, dim=-1)

            mean = mean.unsqueeze(-1)
            variance = variance.unsqueeze(-1)

            x_bar = (x - mean) / torch.sqrt(variance + self.eps)

            return self.alpha * x_bar + self.beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer Normalization for the Medical Assistant".title())
    parser.add_argument("--normalized_shape", type=int, default=768, help="The shape of the input tensor".capitalize())
    parser.add_argument("--eps", type=float, default=1e-05, help="The epsilon value for the normalization".capitalize())
    
    args = parser.parse_args()
    
    image_channels = 3
    image_size = 224
    patch_size = 16
    
    total_patches = (image_size // patch_size) ** 2
    dimension = (image_channels * patch_size * patch_size)
    
    norm = LayerNormalization(normalized_shape=dimension)
    images = torch.randn((image_channels//image_channels, total_patches, dimension))

    assert (
        norm(images).size()
    ) == images.size(), "Layer Normalization is not working properly".capitalize()
