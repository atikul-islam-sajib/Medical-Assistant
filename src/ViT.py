import os
import sys
import torch
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

        # self.transformer = TransformerEncoderBlock(
        #     nhead=self.nhead,
        #     d_model=self.d_model,
        #     dim_feedforward=self.dim_feedforward,
        #     dropout=self.dropout,
        #     activation=self.activation,
        #     layer_norm_eps=self.layer_norm_eps,
        #     bias=self.bias,
        # )

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
    vit = ViT(
        image_channels=1,
        image_size=224,
        patch_size=16,
        encoder_layer=4,
        nhead=8,
        d_model=256,
        dim_feedforward=4 * 256,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
        bias=False,
    )
    
    images = torch.randn((1, 1, 224, 224))
    
    print(vit(images).size())
