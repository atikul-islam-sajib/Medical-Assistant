import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from patch_embedding import PatchEmbedding


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.image_channels = 3
        self.image_size = 224
        self.patch_size = 16
        self.embedding_dimension = 768

        self.patch_embedding = PatchEmbedding(
            image_channels=self.image_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            embedding_dimension=self.embedding_dimension,
        )

        self.images = torch.randn(
            (self.batch_size, self.image_channels, self.image_size, self.image_size)
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


if __name__ == "__main__":
    unittest.main()
