import os
import sys
import torch
import warnings
import torch.nn as nn

sys.path.append("./src/")


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


if __name__ == "__main__":
    multihead_attention = MultiHeadAttentionLayer(nheads=8, dimension=256)
    multihead_attention(x=torch.randn((1, 196, 256)))
