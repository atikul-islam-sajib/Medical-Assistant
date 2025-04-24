import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

try:
    from patch_embedding import PatchEmbedding
    from transformer_block import TransformerEncoderBlock
except ImportError:
    print("Error: Failed to import modules from src/ directory")


class Classifier(nn.Module):
    def __init__(
        self, dimension: int = 768, dropout: float = 0.3, activation: str = "leaky"
    ):
        super(Classifier, self).__init__()

        self.dimension = dimension
        self.dropout = dropout
        self.activation_func = activation

        if self.activation_func == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_func == "leaky":
            self.activation = nn.LeakyReLU(inplace=True)
        elif self.activation_func == "gelu":
            self.activation = nn.GELU()
        elif self.activation_func == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        self.in_features = self.dimension
        self.out_features = self.in_features // 4

        self.layers = []

        for index in range(2):
            self.layers += [
                nn.Linear(in_features=self.in_features, out_features=self.out_features)
            ]

            if index == 0:
                self.layers += [nn.BatchNorm1d(num_features=self.out_features)]
                self.layers += [self.activation]
                self.layers += [nn.Dropout(p=self.dropout)]

            self.in_features = self.out_features
            self.out_features = self.out_features // 4

        self.layers += [
            nn.Sequential(nn.Linear(in_features=self.in_features, out_features=4))
        ]

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            return self.classifier(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classifier for the Medical Assistant".title()
    )
    parser.add_argument(
        "--dimension", type=int, default=768, help="Dimension of the input"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--activation", type=str, default="leaky", help="Activation function"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--target_size", type=int, default=4, help="Target size")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")

    args = parser.parse_args()

    num_of_patches = (args.image_size // args.patch_size) ** 2

    classifier = Classifier(
        dimension=args.dimension,
        dropout=args.dropout,
        activation=args.activation,
    )

    images = torch.randn((16, num_of_patches, args.dimension))
    images = torch.mean(images, dim=1)
    assert (classifier(images).size()) == (
        args.batch_size,
        args.target_size,
    ), "Classifier is not working properly".capitalize()


class ViTWithClassifier(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        target_size: int = 4,
        encoder_layer: int = 4,
        nhead: int = 8,
        d_model: int = 768,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-05,
        bias: bool = False,
    ):
        super(ViTWithClassifier, self).__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.target_size = target_size
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

        self.classifier = Classifier(
            dimension=self.d_model,
            dropout=self.dropout * 2,
            activation=self.activation,
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            x = self.patch_embedding(x)

            for layer in self.transformer:
                x = layer(x)

            x = torch.mean(x, dim=1)
            x = self.classifier(x)

            return x
        
    @staticmethod
    def total_parameters(model: ViTWithClassifier):
        if not isinstance(model, ViTWithClassifier):
            raise TypeError("Input must be a ViTWithClassifier")
        else:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    parser.add_argument("--target_size", type=int, default=4, help="Target size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--display", type=bool, default=True, help="Display the graph")
    
    args = parser.parse_args()

    image_channels = args.image_channels
    image_size = args.image_size
    patch_size = args.patch_size
    target_size = args.target_size
    encoder_layer = args.encoder_layer
    nhead = args.nhead
    d_model = args.d_model
    dim_feedforward = args.dim_feedforward
    dropout = args.dropout
    activation = args.activation
    layer_norm_eps = args.layer_norm_eps
    bias = args.bias

    vit = ViTWithClassifier(
        image_channels=image_channels,
        image_size=image_size,
        patch_size=patch_size,
        target_size=target_size,
        encoder_layer=encoder_layer,
        nhead=nhead,
        d_model=d_model,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        bias=bias,
    )

    images = torch.randn((args.batch_size, image_channels, image_size, image_size))

    num_of_patches = (image_size // patch_size) ** 2

    assert (vit(images).size()) == (
        args.batch_size,
        target_size,
    ), "ViTWithClassifier is not working correctly".capitalize()
    

    if args.display:
        draw_graph(model=vit, input_data=images).visual_graph.render(
            filename="./artifacts/files/ViT", format="png"
        )
        print(
            "Layer Normalization graph has been saved to ./artifacts/files/ViT.png"
        )
