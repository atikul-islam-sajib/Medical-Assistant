import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from layer_normalization import LayerNormalization
from multi_head_attention_layer import MultiHeadAttentionLayer
from feed_forward_neural_network import FeedForwardNeuralNetwork


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        nhead: int = 8,
        d_model: int = 768,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.nheads = nhead
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        self.multi_head_attention = MultiHeadAttentionLayer(
            nheads=self.nheads,
            dimension=self.d_model,
        )

        self.layer_norm1 = LayerNormalization(
            normalized_shape=self.d_model, eps=self.layer_norm_eps
        )
        self.layer_norm2 = LayerNormalization(
            normalized_shape=self.d_model, eps=self.layer_norm_eps
        )
        self.feed_forward_network = FeedForwardNeuralNetwork(
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            residual = x

            x = self.multi_head_attention(x)
            x = self.dropout1(x)
            x = torch.add(x, residual)
            x = self.layer_norm1(x)

            residual = x

            x = self.feed_forward_network(x)
            x = torch.add(x, residual)
            x = self.layer_norm2(x)

            return x
        
    @staticmethod
    def total_parameters(model: TransformerEncoderBlock):
        if not isinstance(model, TransformerEncoderBlock):
            raise TypeError("Input must be a TransformerEncoderBlock")
        else:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer Encoder Block for the Medical Assistant".title()
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of heads in the multi-head attention layer",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=768,
        help="Dimension of the input and output tensors",
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
        help="Epsilon value for layer normalization",
    )
    parser.add_argument(
        "--bias", type=bool, default=False, help="Bias value for the linear layers"
    )
    parser.add_argument("--display", type=bool, default=True, help="Display the graph")

    args = parser.parse_args()

    nhead = args.nhead
    d_model = args.d_model
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    activation = args.activation
    layer_norm_eps = args.layer_norm_eps
    bias = args.bias
    
    image_size = 224
    patch_size = 16
    image_channels = 3

    transformer = TransformerEncoderBlock(
        nhead=nhead,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        bias=bias,
    )

    num_of_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size**2 * image_channels

    images = torch.randn((1, num_of_patches, patch_dim))

    assert (
        transformer(images).size()
    ) == images.size(), "Transformer Encoder Block is not working properly".capitalize()
    

    if args.display:
        draw_graph(model=transformer, input_data=images).visual_graph.render(
            filename="./artifacts/files/transformer", format="png"
        )
        print(
            "Layer Normalization graph has been saved to ./artifacts/files/transformer.png"
        )