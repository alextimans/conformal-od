import os
from torch.utils.data.dataloader import DataLoader

from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import load_coco_json
from detectron2.config import CfgNode
from detectron2.structures import BoxMode

from util.io_file import load_json
from util.util import get_coco_classes, get_bdd_classes
from data.cityscapes import load_cityscapes_instances


def d2_register_dataset(cfg: dict, formatter=None, name=None, logger=None):
    """
    Register a custom dataset with the detectron2 framework.

    Args:
        cfg (dict): config dict
        formatter (Callable, optional): Custom formatting function to bring data 
            into detectron2 dataset representation. Should return list[dict], 
            see also the detectron2 "Register a Dataset" docs. Defaults to None.
        name (str, optional): Dataset name. If None, is inferred from config file.

    Raises:
        ValueError: If no inbuilt data formatter is used, then a formatting function has to be provided, otherwise throw error.
    """
    cfg = CfgNode(cfg)

    meta_name = cfg.DATASETS.DATASET.NAME
    if name is None:
        name = meta_name

    img_dir = os.path.join(cfg.DATASETS.DIR, cfg.DATASETS.DATASET.IMG_DIR)
    ann_file = os.path.join(cfg.DATASETS.DIR, cfg.DATASETS.DATASET.ANN_FILE)
    data_format = cfg.DATASETS.DATASET.FORMAT

    meta = cfg.DATASETS.DATASET.METADATA_FILE
    if meta is None:
        metadata = {}
    else:
        meta_dir, meta_file = os.path.split(meta)
        metadata = load_json(meta_file, os.path.join(cfg.DATASETS.DIR, meta_dir))

    if formatter is None and data_format is not None:
        # detectron2 has inbuilt data formatters/loaders for COCO, LVIS and Pascal VOC
        if data_format == "coco":
            # Uses COCO API under the hood; automatically sets this metadata for COCO data: thing_classes, thing_dataset_id_to_contiguous_id
            formatter = load_coco_json
            metadata["evaluator_type"] = data_format  # Add metadata for inbuilt COCOEvaluator (not used)
            metadata["json_file"] = ann_file
        elif data_format == "bdd":
            formatter = format_bdd
            metadata["thing_classes"] = get_coco_classes()
            metadata["json_file"] = ann_file
        elif data_format == "cityscapes":
            formatter = load_cityscapes_instances
            metadata["thing_classes"] = get_coco_classes()
        else:
            raise ValueError("This data_format is not specified")
    elif formatter is None and data_format is None:
        raise ValueError("If no predefined format need to provide formatting function.")

    # Following along detectron2.data.datasets.register_coco_instances
    assert isinstance(name, str), name
    assert isinstance(ann_file, (str, os.PathLike)), ann_file
    assert isinstance(img_dir, (str, os.PathLike)), img_dir

    DatasetCatalog.register(name, lambda: formatter(ann_file, img_dir, meta_name))
    MetadataCatalog.get(name).set(image_root=img_dir, **metadata)
    if logger is not None:
        logger.info(f"Registered dataset '{name}' with detectron2.")
        logger.info(f"Added metadata for {MetadataCatalog.get(name).as_dict().keys()}.")


def d2_load_dataset_from_cfg(cfg: dict, cfg_model: CfgNode, train: bool = False, logger=None):
    """
    Load a custom dataset registered with detectron2 framework from a config file.

    Args:
        cfg (dict): config dict
        cfg_model (CfgNode): config model in yacs format
        train (bool, optional): Dataloader settings in train mode? Defaults to False.

    Returns:
        torch.utils.data.DataLoader
    """
    cfg = CfgNode(cfg)
    name = cfg.DATASETS.DATASET.NAME

    if train:
        logger.info("Returning train dataloader.")
        cfgt = cfg.DATALOADER.TRAIN
        loader = build_detection_train_loader(
            cfg_model,
            name,
            total_batch_size=cfgt.TOTAL_BATCH_SIZE,
            aspect_ratio_grouping=cfgt.ASPECT_RATIO_GROUPING,
            num_workers=cfgt.NUM_WORKERS,
            collate_fn=cfgt.COLLATE_FN,
        )
        text = "train"
    else:
        logger.info("Returning test dataloader.")
        cfgt = cfg.DATALOADER.TEST
        loader = build_detection_test_loader(
            cfg_model,
            name,  # type: ignore
            batch_size=cfgt.BATCH_SIZE,
            num_workers=cfgt.NUM_WORKERS,
            collate_fn=cfgt.COLLATE_FN,
        )
        text = "test"

    assert isinstance(loader, DataLoader), loader
    logger.info(f"Returning dataloader for dataset '{name}' in '{text}' mode.")
    return loader


def d2_load_dataset_from_dict(
    dataset,
    cfg: dict,
    cfg_model: CfgNode,
    mapper=None,
    sampler=None,
    train: bool = False,
    logger=None,
):
    """
    Load a custom dataset registered with detectron2 framework from explicit params.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable).
        cfg (dict): config dict
        cfg_model (CfgNode): config model in yacs format
        mapper (Callable, optional): A (custom) callable which takes a sample (dict) from
            dataset and returns the format to be consumed by the model.
            If None, defaults to DatasetMapper(cfg, ...)
        sampler (Callable, optional): A sampler based on torch.utils.data.sampler. Sampler that
            produces indices to be applied on the dataset. If None, defaults to
            detectron2.data.samplers.TrainingSampler or InferenceSampler (see src).
        train (bool, optional): Dataloader settings in train mode? Defaults to False.

    Returns:
        torch.utils.data.DataLoader
    """
    cfg = CfgNode(cfg)

    if mapper is None:
        mapper = DatasetMapper(cfg_model, is_train=train)

    if train:
        logger.info("Returning train dataloader.")
        cfgt = cfg.DATALOADER.TRAIN
        loader = build_detection_train_loader(
            dataset=dataset,
            mapper=mapper,
            sampler=sampler,
            total_batch_size=cfgt.TOTAL_BATCH_SIZE,
            aspect_ratio_grouping=cfgt.ASPECT_RATIO_GROUPING,
            num_workers=cfgt.NUM_WORKERS,
            collate_fn=cfgt.COLLATE_FN,
        )
    else:
        logger.info("Returning test dataloader.")
        cfgt = cfg.DATALOADER.TEST
        loader = build_detection_test_loader(
            dataset=dataset,
            mapper=mapper,
            sampler=sampler,
            batch_size=cfgt.BATCH_SIZE,
            num_workers=cfgt.NUM_WORKERS,
            collate_fn=cfgt.COLLATE_FN,
        )

    logger.info(f"Returning dataloader for given dataset in mode {train=}.")
    return loader


def format_bdd(ann_file: str, img_dir: str, meta_name: str):
    """
    Loads the annotation file for a BDD100k dataset partition
    and formats it into a list[dict] digestible for detectron2
    (https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts).

    Original BDD100k classes are mapped to (approx.) corresponding COCO classes,
    and the prediction task is reframed as a COCO detection task with 80 classes.
    Otherwise fine-tune training would be necessary for the model to recognize BDD classes.

    Note: mapping "traffic sign" to "stop sign" will lead to expectedly bad performance
    for that class. "pedestrian" and "rider" are both mapped to "person".
    """

    coco_classes = get_coco_classes()
    bdd_classes = get_bdd_classes()
    # Extra map for some outlier labels
    # https://github.com/bdd100k/bdd100k/blob/master/bdd100k/configs/det.toml
    bdd_extra_map = {
        "other person": "pedestrian",
        "other vehicle": "car",
        "trailer": "truck",
    }
    # Map BDD100k classes to best-fitting COCO classes
    map_to_coco_classes = {
        "pedestrian": "person",
        "rider": "person",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "train": "train",
        "motorcycle": "motorcycle",
        "bicycle": "bicycle",
        "traffic light": "traffic light",
        "traffic sign": "stop sign",
    }
    # All BDD100k images have a fixed size 720 x 1280
    BDD_IMG_HEIGHT, BDD_IMG_WIDTH = 720, 1280

    f = os.path.split(ann_file)
    img_anns = load_json(f[1].split(".")[0], f[0])
    dataset_dicts = []

    for i, v in enumerate(img_anns):
        img = {}

        img["file_name"] = os.path.join(img_dir, v["name"])
        img["image_id"] = i
        img["height"] = BDD_IMG_HEIGHT
        img["width"] = BDD_IMG_WIDTH

        if "labels" not in v.keys():  # 10 degenerate cases in bdd100k_train
            continue

        objs = []
        for anno in v["labels"]:
            box = anno["box2d"]
            cat = anno["category"]
            if cat not in bdd_classes:
                cat = bdd_extra_map[cat]
            cat = map_to_coco_classes[cat]

            objs.append(
                {
                    # https://github.com/scalabel/scalabel/blob/fc694f0d85805a6a79ccf04b0b210fce1ee364af/scalabel/label/transforms.py#L75
                    # transform from scalabel box format to d2 box format
                    "bbox": [
                        round(box["x1"], 2),
                        round(box["y1"], 2),
                        round(box["x2"] + 1, 2),
                        round(box["y2"] + 1, 2),
                    ],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": coco_classes.index(cat),
                }
            )
        img["annotations"] = objs

        dataset_dicts.append(img)
    return dataset_dicts
