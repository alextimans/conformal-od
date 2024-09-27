from pathlib import Path
from torch.nn import Module

from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def d2_build_model(cfg: dict, logger):
    """
    Construct detectron2 model and model config file. 
    Multiple config files are merged to create the final model config file,
    with the custom config file overriding the detectron2 model config file.

    Args:
        cfg (dict): config dict

    Returns:
        cfg_model (CfgNode): Model config file in yacs format
        model (torch.nn.Module): constructed model
    """
    cfg = CfgNode(cfg)
    file = cfg.MODEL.FILE

    # Build cfg_model by merging configs from detectron2 and custom
    cfg_model = get_cfg()
    cfg_model.merge_from_file(  # override d2 base defaults with model
        model_zoo.get_config_file(file)
    )

    if cfg.MODEL.LOCAL_CHECKPOINT:
        cfg_model.MODEL.WEIGHTS = cfg.MODEL.CHECKPOINT_PATH
    else:
        cfg_model.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    logger.info(f"Set model weights path to '{cfg_model.MODEL.WEIGHTS}'.")

    cfg_my_model = cfg.MODEL.CONFIG
    cfg_model.merge_from_other_cfg(  # override d2 model defaults with custom
        cfg_my_model
    )

    model = build_model(cfg_model)  # Builds structure with random params
    assert isinstance(model, Module), model
    logger.info(
        f"Constructed cfg_model file and from it built model '{cfg.MODEL.NAME}'."
    )
    return cfg_model, model


def d2_load_model(cfg_model: CfgNode, model, logger):
    # Load model weights
    DetectionCheckpointer(model).load(cfg_model.MODEL.WEIGHTS)
    logger.info(f"Loaded model weights from checkpoint at '{cfg_model.MODEL.WEIGHTS}'.")


def d2_save_model_checkpt(model, filedir: str, filename: str = "last_checkpt", logger=None):
    # Save model checkpoint
    Path(filedir).mkdir(exist_ok=True, parents=True)
    checkpointer = DetectionCheckpointer(model, save_dir=filedir)
    checkpointer.save(filename)
    logger.info(f"Model checkpointed as '{filename}' at '{filedir}'.")
