import os
import sys
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append("./src/")

from utils import device_init
from helper import helper, Criterion
from ViT import ViTWithClassifier

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        model=None,
        epochs: int = 100,
        lr: float = 0.001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        momentum: float = 0.85,
        adam: bool = True,
        SGD: bool = False,
        l1_regularization: bool = False,
        elasticNet_regularization: bool = False,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.elasticNet_regularization = elasticNet_regularization
        self.device = device
        self.verbose = verbose

        self.init = helper(
            model=self.model,
            lr=self.lr,
            adam=self.adam,
            beta1=self.beta1,
            beta2=self.beta2,
            weight_decay=self.weight_decay,
            SGD=self.SGD,
            momentum=self.momentum,
        )

        self.device = device_init(device=device)

        self.train_dataloader = self.init["dataloader"]["train_dataloader"]
        self.test_dataloader = self.init["dataloader"]["test_dataloader"]
        self.valid_dataloader = self.init["dataloader"]["valid_dataloader"]

        self.optimizer = self.init["optimizer"]
        self.criterion = self.init["criterion"]
        self.classifier = self.init["classifier"]

        self.classifier = self.classifier.to(self.device)
        self.criterion = self.criterion.to(self.device)

        assert self.train_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
        assert self.test_dataloader.__class__ == torch.utils.data.dataloader.DataLoader
        assert self.valid_dataloader.__class__ == torch.utils.data.dataloader.DataLoader

        assert self.classifier.__class__ == ViTWithClassifier
        assert self.criterion.__class__ == Criterion

        if self.adam:
            assert self.optimizer.__class__ == torch.optim.Adam
        elif self.SGD:
            assert self.optimizer.__class__ == torch.optim.SGD
        else:
            raise ValueError("Optimizer not supported".capitalize())

        self.loss = float("inf")

    def l1_regularizer(self, model: ViTWithClassifier):
        if not isinstance(model, ViTWithClassifier):
            raise ValueError("Model must be a ViTWithClassifier".capitalize())
        return 0.01 * sum(
            torch.norm(input=params, p=1) for params in model.parameters()
        )

    def elasticNet_regularizer(self, model: ViTWithClassifier):
        if not isinstance(model, ViTWithClassifier):
            raise ValueError("Model must be a ViTWithClassifier".capitalize())
        return 0.01 * sum(
            torch.norm(input=params, p=1) for params in model.parameters()
        ) + 0.01 * sum(torch.norm(input=params, p=2) for params in model.parameters())

    def saved_checkpoints(self, train_loss: float, epoch: int = 1):
        if not isinstance(train_loss, float):
            raise ValueError("Train Loss must be a tensor".capitalize())

        if train_loss < self.loss:
            self.loss = train_loss
            torch.save(
                {
                    "train_loss": self.loss,
                    "epoch": self.epochs,
                    "model_state_dict": self.classifier.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join("./artifacts/checkpoints/best_model", "best_model.pth"),
            )

        torch.save(
            self.classifier.state_dict(),
            os.path.join("./artifacts/checkpoints/train_models", f"model{epoch}.pth"),
        )

    def update_training(self, predicted: torch.Tensor, actual: torch.Tensor):
        if not isinstance(predicted, torch.Tensor) and isinstance(actual, torch.Tensor):
            raise ValueError("Predicted and Actual must be tensors".capitalize())

        self.optimizer.zero_grad()

        predicted_loss = self.criterion(predicted, actual)

        if self.l1_regularization:
            predicted_loss += self.l1_regularizer(self.classifier)
        elif self.elasticNet_regularization:
            predicted_loss += self.elasticNet_regularizer(self.classifier)

        predicted_loss.backward()

        self.optimizer.step()

        return predicted_loss.item()

    def display(self, **kwargs):
        epoch = kwargs["epoch"]
        train_loss = kwargs["train_loss"]
        valid_loss = kwargs["valid_loss"]
        train_accuracy = kwargs["train_accuracy"]
        valid_accuracy = kwargs["valid_accuracy"]

        print(
            "Epochs - [{}/{}] - train_loss: {:.4f} - test_loss: {:.4f} - train_accuracy: {:.4f} - test_accuracy: {:.4f}".format(
                epoch,
                self.epochs,
                train_loss,
                valid_loss,
                train_accuracy,
                valid_accuracy,
            )
        )

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Training Medical-Assistant"):
            train_loss = []
            valid_loss = []
            total_train_predicted_labels = []
            total_valid_predicted_labels = []
            total_train_actual_labels = []
            total_valid_actual_labels = []

            for images, labels in self.train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                predicted = self.classifier(images)

                train_loss.append(
                    self.update_training(predicted=predicted, actual=labels)
                )
                predicted = torch.argmax(input=predicted, dim=1)
                predicted = predicted.detach().cpu().numpy()

                total_train_predicted_labels.append(predicted)
                total_train_actual_labels.append(labels.detach().cpu().numpy())

            for images, labels in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                predicted = self.classifier(images)

                valid_loss.append(self.criterion(predicted, labels).item())

                predicted = torch.argmax(input=predicted, dim=1)
                predicted = predicted.detach().cpu().numpy()

                total_valid_predicted_labels.append(predicted)
                total_valid_actual_labels.append(labels.detach().cpu().numpy())

            train_accuracy = accuracy_score(
                np.concatenate(total_train_predicted_labels),
                np.concatenate(total_train_actual_labels),
            )
            valid_accuracy = accuracy_score(
                np.concatenate(total_valid_predicted_labels),
                np.concatenate(total_valid_actual_labels),
            )

            self.display(
                epoch=epoch + 1,
                train_loss=np.mean(train_loss),
                valid_loss=np.mean(valid_loss),
                train_accuracy=train_accuracy,
                valid_accuracy=valid_accuracy,
            )

            self.saved_checkpoints(train_loss=np.mean(train_loss), epoch=epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for the Medical Assistant")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.85, help="Momentum for SGD")
    parser.add_argument("--adam", type=bool, default=True, help="Use Adam optimizer")
    parser.add_argument("--SGD", type=bool, default=False, help="Use SGD optimizer")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")

    args = parser.parse_args()

    trainer = Trainer(
        model=None,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        adam=args.adam,
        SGD=args.SGD,
        device=args.device,
        verbose=args.verbose,
    )

    trainer.train()
