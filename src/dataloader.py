import os
import cv2
import sys
import torch
import zipfile
import argparse
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./src/")

from utils import dump_files


class Loader:
    def __init__(
        self,
        image_path: str = "./data/raw",
        image_channels: int = 1,
        image_size: int = 224,
        batch_size: int = 64,
        split_size: float = 0.25,
    ):
        self.image_path = image_path
        self.image_channels = image_channels
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.train_images = []
        self.train_labels = []
        self.valid_images = []
        self.valid_labels = []

    def unzip_folder(self):
        if not os.path.exists("./data/processed"):
            os.makedirs("./data/processed")

        with zipfile.ZipFile(file=self.image_path, mode="r") as file:
            file.extractall(path="./data/processed")

        print("""Extracted file saved in the "./data/processed" folder""")

    def split_dataset(self, **kwargs):
        X = kwargs["X"]
        y = kwargs["y"]

        if not isinstance(X, list) and not isinstance(y, list):
            raise ValueError("Invalid data type".capitalize())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def image_transforms(self, type: str = "RGB"):
        if type == "RGB":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def extract_features(self):
        train_path = "./data/processed/Training"
        valid_path = "./data/processed/Testing"

        class_names = ["glioma", "meningioma", "pituitary", "notumor"]

        for path in [train_path, valid_path]:
            for label in class_names:
                image_path = os.path.join(path, label)
                for image in tqdm(
                    os.listdir(image_path), desc="Extracting images".title()
                ):
                    single_image_path = os.path.join(image_path, image)
                    if not single_image_path.endswith(("png", "jpg", "jpeg")):
                        raise ValueError("Invalid image format")

                    image = cv2.imread(single_image_path)
                    image = Image.fromarray(image)
                    image = self.image_transforms(
                        "GRAY" if self.image_channels == 1 else "RGB"
                    )(image)

                    if path == train_path:
                        self.train_images.append(image)
                        self.train_labels.append(class_names.index(label))
                    else:
                        self.valid_images.append(image)
                        self.valid_labels.append(class_names.index(label))

        assert len(self.train_images) == len(self.train_labels)
        assert len(self.valid_images) == len(self.valid_labels)

        train_dataset = self.split_dataset(X=self.train_images, y=self.train_labels)

        return {
            "X_train": torch.stack(train_dataset["X_train"][:200]).float(),
            "X_test": torch.stack(train_dataset["X_test"][:200]).float(),
            "y_train": torch.tensor(train_dataset["y_train"][:50], dtype=torch.long),
            "y_test": torch.tensor(train_dataset["y_test"][:50], dtype=torch.long),
            "valid_images": torch.stack(self.valid_images[:20]).float(),
            "valid_labels": torch.tensor(self.valid_labels[:20], dtype=torch.long),
        }

    def create_dataloader(self):
        dataset = self.extract_features()

        train_dataloader = DataLoader(
            dataset=list(zip(dataset["X_train"], dataset["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            dataset=list(zip(dataset["X_test"], dataset["y_test"])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            dataset=list(zip(dataset["valid_images"], dataset["valid_labels"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for value, filename in tqdm(
            [
                (train_dataloader, "train_dataloader.pkl"),
                (test_dataloader, "test_dataloader.pkl"),
                (valid_dataloader, "valid_dataloader.pkl"),
            ],
            desc="Saving dataloaders".title(),
        ):
            dump_files(
                value=value, filename=os.path.join("./data/processed/", filename)
            )

        print("Files saved in the folder ./data/processed/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataloader for the Medical Assistant Task".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/raw/dataset.zip",
        help="Path to the dataset".capitalize(),
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=1,
        help="Number of image channels".capitalize(),
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Image size".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size".capitalize()
    )
    parser.add_argument(
        "--split_size", type=float, default=0.30, help="Split size".capitalize()
    )

    args = parser.parse_args()

    image_path = args.image_path
    image_channels = args.image_channels
    image_size = args.image_size
    batch_size = args.batch_size
    split_size = args.split_size

    loader = Loader(
        image_path=image_path,
        image_channels=image_channels,
        image_size=image_size,
        batch_size=batch_size,
        split_size=split_size,
    )

    loader.unzip_folder()
    loader.create_dataloader()
