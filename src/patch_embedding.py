import os
import sys
import torch
import warnings
import argparse
import torch.nn as nn
from torchview import draw_graph

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
                self.image_channels * self.patch_size * self.patch_size
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
    parser = argparse.ArgumentParser(
        description="Patch Embedding for the Medical Assistant Task".capitalize()
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=3,
        help="The number of channels in the input image".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="The size of the input image".capitalize(),
    )
    parser.add_argument(
        "--patch_size", type=int, default=16, help="The size of the patch".capitalize()
    )
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        default=512,
        help="The dimension of the embedding".capitalize(),
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display the graph".capitalize()
    )

    args = parser.parse_args()

    image_channels = args.image_channels
    image_size = args.image_size
    patch_size = args.patch_size
    embedding_dimension = args.embedding_dimension

    patch_embedding = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embedding_dimension=embedding_dimension,
    )

    image = torch.randn((1, image_channels, image_size, image_size))

    assert (patch_embedding(image).size()) == (
        1,
        (image_size // patch_size) ** 2,
        embedding_dimension,
    ), "Patch embedding is not working correctly".capitalize()

    if args.display:
        draw_graph(model=patch_embedding, input_data=image).visual_graph.render(
            filename="./artifacts/files/patch_embedding", format="png"
        )
        print(
            "Layer Normalization graph has been saved to ./artifacts/files/patch_embedding.png"
        )
