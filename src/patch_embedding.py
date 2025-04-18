import os
import sys
import torch
import warnings
import torch.nn as nn

sys.path.append("./src/")


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int = 512):
        super(PositionalEncoding, self).__init__()
        self.dimension = dimension

        self.encoding = nn.Parameter(
            torch.randn(
                size=(
                    self.dimension // self.dimension,
                    self.dimension // self.dimension,
                    self.dimension,
                ),
                requires_grad=True,
            )
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            return x + self.encoding


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        embedding_dimension: int = 512,
    ):
        super(PatchEmbedding, self).__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dimension = embedding_dimension

        self.total_patches = (self.image_size // self.patch_size) ** 2

        if self.embedding_dimension is None:
            warnings.warn(
                "Embedding dimension not specified. Using the default value calculated as: image_channels × patch_size × patch_size."
            )
            self.embedding_dimension = (
                self.image_channels**self.patch_size * self.patch_size
            )

        self.projection = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=self.embedding_dimension,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=self.patch_size // self.patch_size,
            bias=False,
        )
        self.encoding = PositionalEncoding(dimension=self.embedding_dimension)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            x = self.projection(x)
            x = x.view(x.size(0), x.size(-1) * x.size(-2), x.size(1))
            x = self.encoding(x)
            return x


if __name__ == "__main__":
    patch_embedding = PatchEmbedding(
        image_size=224,
        patch_size=16,
        embedding_dimension=512,
    )

    image = torch.randn((1, 3, 224, 224))

    print(patch_embedding(image).size())
