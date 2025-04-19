import os
import sys
import math
import torch
import argparse
import warnings
import torch.nn as nn

sys.path.append("./src/")


def scaled_dot_product(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if (
        not isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        raise TypeError("All inputs must be torch.Tensor".capitalize())

    key = key.transpose(-2, -1)
    scores = torch.matmul(query, key) / math.sqrt(key.size(-1))
    scores = torch.softmax(scores, dim=-1)
    attention = torch.matmul(scores, value)
    return attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, nheads: int = 6, dimension: int = 768):
        super(MultiHeadAttentionLayer, self).__init__()
        self.nheads = nheads
        self.dimension = dimension

        assert (
            self.dimension % self.nheads == 0
        ), "Dimension must be divisible by number of heads".capitalize()

        warnings.warn(
            "Invalid number of dimensions provided. To avoid errors, ensure the dimension is calculated as: in_channels × patch_size × patch_size."
        )

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=False
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor".capitalize())
        else:
            QKV = self.QKV(x)
            query, key, value = torch.chunk(input=QKV, chunks=3, dim=-1)
            assert (
                query.size() == key.size() == value.size()
            ), "Query, key, and value must have the same size".capitalize()

            query = query.view(
                query.size(0), query.size(1), self.nheads, query.size(-1) // self.nheads
            )
            key = key.view(
                key.size(0), key.size(1), self.nheads, key.size(-1) // self.nheads
            )
            value = value.view(
                value.size(0), value.size(1), self.nheads, value.size(-1) // self.nheads
            )

            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            attention = scaled_dot_product(query=query, key=key, value=value)
            attention = attention.view(
                attention.size(0),
                attention.size(-2),
                attention.size(1),
                attention.size(-1),
            )
            attention = attention.view(
                attention.size(0),
                attention.size(1),
                attention.size(2) * attention.size(3),
            )

            assert (
                attention.size() == x.size()
            ), "Attention output must have the same size as input".capitalize()

            return attention


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi Head Attention Layer for the Medical Assistant".title()
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=6,
        help="Number of heads for the multi head attention layer",
    )
    parser.add_argument(
        "--dimension", type=int, default=768, help="Dimension of the input tensor"
    )
    args = parser.parse_args()

    nheads = args.nheads
    dimension = args.dimension

    images = torch.randn((dimension // dimension, 196, dimension))

    multihead_attention = MultiHeadAttentionLayer(nheads=nheads, dimension=dimension)
    assert (
        multihead_attention(x=images)
    ).size() == images.size(), (
        "Multi head attention output must have the same size as input".capitalize()
    )
