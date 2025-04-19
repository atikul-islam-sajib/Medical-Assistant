import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn

sys.path.append("./src/")

from patch_embedding import PatchEmbedding
from transformer_block import TransformerEncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
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
        self.image_channels = image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_layer = encoder_layer
        self.nhead = nhead
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        self.layers = []

        self.patch_embedding = PatchEmbedding(
            image_channels=self.image_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            embedding_dimension=self.d_model,
        )

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    nhead=self.nhead,
                    d_model=self.d_model,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                    layer_norm_eps=self.layer_norm_eps,
                    bias=self.bias,
                )
                for _ in tqdm(
                    range(self.encoder_layer), desc="Transformer Block".title()
                )
            ]
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            x = self.patch_embedding(x)

            for layer in self.transformer:
                x = layer(x)

            return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Model for the Medical Assistant")
    parser.add_argument(
        "--image_channels", type=int, default=3, help="Number of image channels"
    )
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument(
        "--encoder_layer", type=int, default=4, help="Number of encoder layers"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of heads in the multi-head attention",
    )
    parser.add_argument(
        "--d_model", type=int, default=768, help="Dimension of the model"
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of the feedforward network",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--activation", type=str, default="gelu", help="Activation function"
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=1e-05,
        help="Layer normalization epsilon",
    )
    parser.add_argument("--bias", type=bool, default=False, help="Bias")
    args = parser.parse_args()

    image_channels = args.image_channels
    image_size = args.image_size
    patch_size = args.patch_size
    encoder_layer = args.encoder_layer
    nhead = args.nhead
    d_model = args.d_model
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    activation = args.activation
    layer_norm_eps = args.layer_norm_eps
    bias = args.bias

    vit = ViT(
        image_channels=image_channels,
        image_size=image_size,
        patch_size=patch_size,
        encoder_layer=encoder_layer,
        nhead=nhead,
        d_model=d_model,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        bias=bias,
    )

    images = torch.randn((image_channels//image_channels, image_channels, image_size, image_size))

    assert (vit(images).size()) == images.size(), "ViT is not working properly".capitalize()
