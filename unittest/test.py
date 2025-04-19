import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from patch_embedding import PatchEmbedding
from multi_head_attention_layer import MultiHeadAttentionLayer


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.image_channels = 1
        self.image_size = 224
        self.patch_size = 16
        self.nheads = 8
        self.embedding_dimension = 256

        self.patch_embedding = PatchEmbedding(
            image_channels=self.image_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            embedding_dimension=self.embedding_dimension,
        )

        self.images = torch.randn(
            (self.batch_size, self.image_channels, self.image_size, self.image_size)
        )

        self.multihead_attention = MultiHeadAttentionLayer(
            nheads=self.nheads, dimension=self.embedding_dimension
        )

    def test_patch_embedding(self):
        self.assertEqual(
            self.patch_embedding(self.images).size(),
            (
                self.batch_size,
                (self.image_size // self.patch_size) ** 2,
                self.embedding_dimension,
            ),
        )

    def test_multi_head_attention(self):
        self.assertEqual(
            self.multihead_attention(x=self.patch_embedding(x=self.images)).size(),
            self.patch_embedding(x=self.images).size(),
        )


if __name__ == "__main__":
    unittest.main()
