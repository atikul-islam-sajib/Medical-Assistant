import os
import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from helper import helper
from helper import Criterion
from ViT import ViTWithClassifier
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

        self.init = helper(
            model=None,
            lr=1e-4,
            adam=True,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            SGD=False,
            momentum=0.9,
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
        
    def test_helper(self):
        train_dataloader = self.init["dataloader"]["train_dataloader"]
        test_dataloader = self.init["dataloader"]["test_dataloader"]
        valid_dataloader = self.init["dataloader"]["valid_dataloader"]

        classifier = self.init["classifier"]

        criterion = self.init["criterion"]

        assert train_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
        assert test_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
        assert valid_dataloader.__class__ == torch.utils.data.dataloader.DataLoader

        assert classifier.__class__ == ViTWithClassifier
        assert criterion.__class__ == Criterion


if __name__ == "__main__":
    unittest.main()
