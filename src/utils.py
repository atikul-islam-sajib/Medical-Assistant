import os
import sys
import yaml
import torch
import joblib
import torch.nn as nn

sys.path.append("./src/")


def config_files():
    with open("./train_config.yml", mode="r") as file:
        return yaml.safe_load(file)


def dump_files(value=None, filename=None):
    if (value is None) and (filename is None):
        raise ValueError("Either values or filename must be provided".capitalize())
    else:
        joblib.dump(value=value, filename=filename)


def load_files(filename: str = None):
    if filename is None:
        raise ValueError("Filename must be provided".capitalize())
    else:
        return joblib.load(filename=filename)


def device_init(device: str = "cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
