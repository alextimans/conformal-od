import os
import random
import numpy
import torch

from detectron2.config import CfgNode

"""
Different utility functions used in the project:
- set_wd: set working directory
- set_seed: set seed for reproducibility
- set_device: set device for training
- get_coco_classes: get COCO classes
- get_bdd_classes: get BDD classes
- get_bdd_as_coco_classes: get BDD classes mapped to COCO classes
- get_cityscapes_classes: get Cityscapes classes
- get_cityscapes_as_coco_classes: get Cityscapes classes mapped to COCO classes
- get_selected_coco_classes: get selected COCO classes
- get_map_from_detr_to_coco: get mapping from DETR classes to COCO classes
"""


def set_wd(target_wd: str):
    curr_wd = os.getcwd()
    if curr_wd != target_wd:
        print(f"Changing wd: {curr_wd} -> {target_wd}")
        os.chdir(target_wd)
    else:
        print("Already in target wd.")


def set_seed(seed: int, logger, verbose: bool = True):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # https://pytorch.org/docs/stable/backends.html
    torch.backends.cudnn.benchmark = True  # Randomness but better runtime
    torch.backends.cudnn.deterministic = False  # Randomness but better runtime
    if verbose:
        logger.info(f"Setting {seed=}.")


def set_device(cfg: dict, device: str, logger):
    cfg = CfgNode(cfg)
    logger.info(f"Requested {device=}.")
    gpu_avail = torch.cuda.is_available()

    if (device == "cpu") and (not gpu_avail):
        pass
    elif (device == "cpu") and gpu_avail:
        logger.info("'cpu' requested but 'cuda' is available.")
    elif (device == "cuda") and (not gpu_avail):
        logger.info("'cuda' requested but not available, using 'cpu' instead.")
        device = "cpu"
    elif (device == "cuda") and gpu_avail:
        logger.info(f"Using 'cuda', {torch.cuda.device_count()} devices available.")
        logger.info(f"Using GPU {torch.cuda.get_device_name()}.")

    cfg.MODEL.DEVICE = device
    cfg.MODEL.CONFIG.MODEL.DEVICE = device
    return cfg, device


def get_coco_classes():
    return [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def get_bdd_classes():
    # https://doc.bdd100k.com/format.html#categories
    return [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]


def get_bdd_as_coco_classes():
    # COCO classes to which BDD was mapped, with corresponding class idx
    return {
        "person": 0,
        "car": 2,
        "truck": 7,
        "bus": 5,
        "train": 6,
        "motorcycle": 3,
        "bicycle": 1,
        "traffic light": 9,
        "stop sign": 11,
    }


def get_cityscapes_classes():
    # https://www.cityscapes-dataset.com/dataset-overview/
    # classes with instance annotations
    return ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def get_cityscapes_as_coco_classes():
    # COCO classes to which cityscapes was mapped, with corresponding class idx
    return {
        "person": 0,
        "car": 2,
        "truck": 7,
        "bus": 5,
        "train": 6,
        "motorcycle": 3,
        "bicycle": 1,
    }


def get_selected_coco_classes():
    # selection of COCO classes, based on a
    # balance of class relevance and availability across COCO, BDD100k, Cityscapes
    return {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def get_map_from_detr_to_coco():
    # DETR model returns 91 classes of which some are "N/A"
    return {
        0: -1,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        12: -1,
        13: 11,
        14: 12,
        15: 13,
        16: 14,
        17: 15,
        18: 16,
        19: 17,
        20: 18,
        21: 19,
        22: 20,
        23: 21,
        24: 22,
        25: 23,
        26: -1,
        27: 24,
        28: 25,
        29: -1,
        30: -1,
        31: 26,
        32: 27,
        33: 28,
        34: 29,
        35: 30,
        36: 31,
        37: 32,
        38: 33,
        39: 34,
        40: 35,
        41: 36,
        42: 37,
        43: 38,
        44: 39,
        45: -1,
        46: 40,
        47: 41,
        48: 42,
        49: 43,
        50: 44,
        51: 45,
        52: 46,
        53: 47,
        54: 48,
        55: 49,
        56: 50,
        57: 51,
        58: 52,
        59: 53,
        60: 54,
        61: 55,
        62: 56,
        63: 57,
        64: 58,
        65: 59,
        66: -1,
        67: 60,
        68: -1,
        69: -1,
        70: 61,
        71: -1,
        72: 62,
        73: 63,
        74: 64,
        75: 65,
        76: 66,
        77: 67,
        78: 68,
        79: 69,
        80: 70,
        81: 71,
        82: 72,
        83: -1,
        84: 73,
        85: 74,
        86: 75,
        87: 76,
        88: 77,
        89: 78,
        90: 79
    }