import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config_files
from tester import Tester
from trainer import Trainer
from dataloader import Loader


def cli():
    parser = argparse.ArgumentParser(description="Medical Assistant CLI".capitalize())
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/raw/dataset.zip",
        help="Path to the dataset".capitalize(),
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=config_files()["dataloader"]["image_channels"],
        help="Number of image channels".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config_files()["dataloader"]["image_size"],
        help="Image size".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config_files()["dataloader"]["batch_size"],
        help="Batch size".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config_files()["dataloader"]["split_size"],
        help="Split size".capitalize(),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config_files()["Trainer"]["epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config_files()["Trainer"]["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config_files()["Trainer"]["beta1"],
        help="Beta1 for Adam",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config_files()["Trainer"]["beta2"],
        help="Beta2 for Adam",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config_files()["Trainer"]["weight_decay"],
        help="Weight decay",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config_files()["Trainer"]["momentum"],
        help="Momentum for SGD",
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config_files()["Trainer"]["adam"],
        help="Use Adam optimizer",
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config_files()["Trainer"]["SGD"],
        help="Use SGD optimizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config_files()["Trainer"]["device"],
        help="Device to use",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config_files()["Trainer"]["verbose"],
        help="Verbose mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config_files()["Tester"]["dataset"],
        help="Dataset to be used for testing",
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            image_path=args.image_path,
            image_channels=args.image_channels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        loader.unzip_folder()
        loader.create_dataloader()

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
            l1_regularization=False,
            elasticNet_regularization=False,
        )

        trainer.train()

    elif args.test:
        tester = Tester(
            dataset=args.dataset,
            device=args.device,
        )
        tester.test()


if __name__ == "__main__":
    cli()
