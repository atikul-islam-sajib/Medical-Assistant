import os
import sys
import yaml
import torch
import joblib

sys.path.append("./src/")


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
