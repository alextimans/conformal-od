import os
import yaml
import json
import torch
from pathlib import Path
import numpy as np

from detectron2.config import CfgNode


def d2_model_config_to_yaml(cfg: CfgNode, filename: str, filedir: str):
    # Save a detectron2 model config stored in a CfgNode to a yaml file
    Path(filedir).mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(filedir, filename + ".yaml")
    with open(filepath, "w") as file:
        f = cfg.dump()
        file.write("# Empty dump" if f is None else f)
    print(f"Written {filename} to {filepath}.")


def load_yaml(filename: str, filedir: str, to_yacs: bool = False):
    filepath = os.path.join(filedir, filename + ".yaml")
    with open(filepath, "r") as file:
        try:
            parsed_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            parsed_file = {}
            print(exc)

    if to_yacs:
        parsed_file = CfgNode(parsed_file)

    print(f"Loaded YAML file from {filepath} into {parsed_file.__class__}.")
    return parsed_file


def save_txt(file, filename: str, filedir: str, **kwargs):
    Path(filedir).mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(filedir, filename + ".txt")
    np.savetxt(filepath, file, **kwargs)
    print(f"Written {filename} to {filepath}.")


def save_json(file, filename: str, filedir: str, **kwargs):
    Path(filedir).mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(filedir, filename + ".json")
    with open(filepath, "w") as f:
        json.dump(file, f, **kwargs)
    print(f"Written {filename} to {filepath}.")


def load_json(filename: str, filedir: str, **kwargs):
    filepath = os.path.join(filedir, filename + ".json")
    with open(filepath, "r") as f:
        try:
            parsed_file = json.load(f, **kwargs)
        except json.JSONDecodeError as exc:
            parsed_file = []
            print(exc)
    print(f"Loaded JSON file from {filepath}.")
    return parsed_file


def save_tensor(file, filename: str, filedir: str, **kwargs):
    Path(filedir).mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(filedir, filename + ".pt")
    torch.save(file, filepath, **kwargs)
    print(f"Written {filename} to {filepath}.")


def load_tensor(filename: str, filedir: str, **kwargs):
    filepath = os.path.join(filedir, filename + ".pt")
    file = torch.load(filepath, **kwargs)
    print(f"Loaded torch tensor from {filepath}.")
    return file
