import os
import sys
import torch
import argparse
import warnings
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.append("./src/")

from utils import device_init, load_files, config_files
from ViT import ViTWithClassifier

warnings.filterwarnings("ignore")


class Tester:
    def __init__(self, dataset: str = "test", device: str = "cuda"):
        self.dataset = dataset
        self.device = device

        self.device = device_init(device=device)

    def load_dataset(self):
        path = "./data/processed/"
        if self.dataset == "test":
            test_dataloader = os.path.join(path, "test_dataloader.pkl")
            test_dataloader = load_files(filename=test_dataloader)

            return test_dataloader

        elif self.dataset == "valid":
            val_dataloader = os.path.join(path, "valid_dataloader.pkl")
            val_dataloader = load_files(filename=val_dataloader)

            return val_dataloader

        else:
            raise ValueError("Dataset must be either 'test' or 'valid'")

    def load_model(self):
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
        ).to(self.device)

        path = "./artifacts/checkpoints/best_model/best_model.pth"
        model = torch.load(path)

        state_dict = model["model_state_dict"]
        classifier.load_state_dict(state_dict=state_dict)

        return classifier

    def test(self):
        dataset = self.load_dataset()
        classifier = self.load_model()

        classifier.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in dataset:

                images = images.to(device=self.device)
                labels = labels.to(device=self.device)

                outputs = classifier(images)

                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        print("Accuracy: ", accuracy_score(y_true, y_pred))
        print("Precision: ", precision_score(y_true, y_pred, average="weighted"))
        print("Recall: ", recall_score(y_true, y_pred, average="weighted"))
        print("F1 Score: ", f1_score(y_true, y_pred, average="weighted"), "\n")

        print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred), "\n")
        print("Classification Report: \n", classification_report(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tester code for the evaluation".title()
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config_files()["Tester"]["dataset"],
        help="Dataset to be used for testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config_files()["Tester"]["device"],
        help="Device to be used for testing",
    )

    args = parser.parse_args()

    dataset = args.dataset
    device = args.device

    tester = Tester(dataset=dataset, device=device)
    tester.test()
