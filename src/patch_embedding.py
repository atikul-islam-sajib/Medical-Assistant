import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embedding_dimension: int = 512,
    ):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dimension = embedding_dimension
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            pass
        
        
if __name__ == "__main__":
    patch_embedding = PatchEmbedding(
        image_size=224,
        patch_size=16,
        embedding_dimension=512,
    )