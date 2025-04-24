import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append("./src/")

from ViT import ViTWithClassifier
from utils import load_files, config_files


class Criterion(nn.Module):
    def __init__(self, loss_function: str = "cross_entropy", reduction: str = "mean"):
        super(Criterion, self).__init__()
        self.loss_function = loss_function
        self.reduction = reduction

        if loss_function == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)
        elif loss_function == "cross_entropy_with_logits":
            self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)
        else:
            raise ValueError("Invalid loss function")

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not isinstance(
            predicted, torch.Tensor
        ):
            raise ValueError("Inputs must be torch.Tensors")

        return self.criterion(predicted, actual)


def load_dataloader():
    dataloader_path = "./data/processed"
    train_dataloader = os.path.join(dataloader_path, "train_dataloader.pkl")
    test_dataloader = os.path.join(dataloader_path, "test_dataloader.pkl")
    valid_dataloader = os.path.join(dataloader_path, "valid_dataloader.pkl")

    train_dataloader = load_files(filename=train_dataloader)
    test_dataloader = load_files(filename=test_dataloader)
    valid_dataloader = load_files(filename=valid_dataloader)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "valid_dataloader": valid_dataloader,
    }


def helper(**kwargs):
    model = kwargs["model"]
    lr: float = float(kwargs["lr"])
    weight_decay: float = float(kwargs["weight_decay"])
    adam: bool = kwargs["adam"]
    beta1: float = float(kwargs["beta1"])
    beta2: float = float(kwargs["beta2"])
    SGD: bool = kwargs["SGD"]
    momentum: float = float(kwargs["momentum"])

    if model is None:
        classifier = ViTWithClassifier(
            image_channels=config_files()["dataloader"]["image_channels"],
            image_size=config_files()["dataloader"]["image_size"],
            patch_size=config_files()["ViT"]["patch_size"],
            target_size=config_files()["dataloader"]["target_size"],
            encoder_layer=config_files()["ViT"]["encoder_layer"],
            nhead=config_files()["ViT"]["nhead"],
            d_model=config_files()["ViT"]["d_model"],
            dim_feedforward=config_files()["ViT"]["dim_feedforward"],
            dropout=config_files()["ViT"]["dropout"],
            activation=config_files()["ViT"]["activation"],
            layer_norm_eps=float(config_files()["ViT"]["layer_norm_eps"]),
            bias=config_files()["ViT"]["bias"],
        )
    else:
        classifier = model

    if adam:
        optimizer = optim.Adam(
            params=classifier.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    elif SGD:
        optimizer = optim.SGD(
            params=classifier.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("Optimizer not found, use 'adam' or 'SGD'")
    try:
        dataloader = load_dataloader()
    except Exception as e:
        raise ValueError(
            "Dataloader not found, use 'load_dataloader' to load dataloader"
        )

    try:
        criterion = Criterion(loss_function="cross_entropy")
    except Exception as e:
        raise ValueError("Criterion not found, use 'Criterion' to load criterion")

    return {
        "classifier": classifier,
        "optimizer": optimizer,
        "criterion": criterion,
        "dataloader": dataloader,
    }


if __name__ == "__main__":
    init = helper(
        model=None,
        lr=1e-4,
        adam=True,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
        SGD=False,
        momentum=0.9,
    )

    train_dataloader = init["dataloader"]["train_dataloader"]
    test_dataloader = init["dataloader"]["test_dataloader"]
    valid_dataloader = init["dataloader"]["valid_dataloader"]

    classifier = init["classifier"]

    criterion = init["criterion"]

    assert train_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
    assert test_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
    assert valid_dataloader.__class__ == torch.utils.data.dataloader.DataLoader

    assert classifier.__class__ == ViTWithClassifier
    assert criterion.__class__ == Criterion
