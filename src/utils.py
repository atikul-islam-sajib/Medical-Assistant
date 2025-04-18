import os
import sys
import yaml
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
