"""
This script contains the training loop for training the quantile regression head
on top of a pre-trained object detection model. The script is based on the
Detectron2 training classes and functions as far as possible.
"""

import sys
import os
import argparse
import wandb
from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    EventWriter,
    get_event_storage,
)

from util import util, io_file
from data.data_loader import d2_load_dataset_from_dict, d2_register_dataset
from model.qr_head import QuantileROIHead


class WAndBWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    Adapted from https://github.com/facebookresearch/detectron2/issues/774
    """

    def __init__(self, wandb_obj=None, window_size: int = 20):
        self.wandb = wandb_obj
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        stats = {}
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            stats[k.replace("/", "-")] = v[0]
        self.wandb.log(stats, step=storage.iter)

        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

    def close(self):
        pass


class Trainer(DefaultTrainer):
    """
    see also DefaultTrainer, after which we model this class
    """

    def __init__(self, cfg_all, cfg, filedir, wandb_run, logger):
        self.cfg_all = cfg_all
        self.filedir = filedir
        self.wandb_run = wandb_run
        self.logger = logger
        super().__init__(cfg)

    def build_model(self, cfg):
        model = build_model(cfg)
        self.logger.info(
            f"From cfg_model file built model '{self.cfg_all.MODEL.NAME}'."
        )
        self.freeze_params(model)  # design choice: which params to freeze
        self.logger.info(f"Box head architecture: {model.roi_heads}")
        return model

    def freeze_params(self, model):
        freeze = self.cfg_all.QUANTILE_REGRESSION.FREEZE_PARAMS
        names = [name for name, _ in model.named_parameters()]

        if freeze == "all":
            unfreeze = [n for n in names if "bbox_pred_" in n]
        elif freeze == "head":
            unfreeze = [n for n in names if "box_predictor" in n]
        else:
            unfreeze = []

        for name, param in model.named_parameters():
            if name in unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.logger.info(f"Unfrozen params: {unfreeze}.")

    def build_train_loader(self, cfg):
        d2_register_dataset(self.cfg_all, logger=self.logger)
        self.data_list = get_detection_dataset_dicts(
            self.cfg_all.DATASETS.DATASET.NAME, filter_empty=True
        )
        self.metadata = MetadataCatalog.get(
            self.cfg_all.DATASETS.DATASET.NAME
        ).as_dict()
        return d2_load_dataset_from_dict(
            self.data_list, self.cfg_all, cfg, train=True, logger=self.logger
        )

    def build_writers(self):
        output_dir = self.cfg.OUTPUT_DIR
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
        ]
        if self.wandb_run is not None:
            writers.append(WAndBWriter(self.wandb_run))
        return writers


def setup(args):
    # load config
    cfg = io_file.load_yaml(args.config_f, args.config_path, to_yacs=True)
    cfg.merge_from_list(args.opts)  # override custom with cli args

    # setup checkpoint folder
    if args.file_name_prefix is not None:
        file_name_prefix = args.file_name_prefix
    else:
        file_name_prefix = f"{cfg.MODEL.ID}{args.file_name_suffix}"
    filedir = os.path.join(cfg.PROJECT.CHECKPOINT_DIR, file_name_prefix)
    Path(filedir).mkdir(exist_ok=True, parents=True)

    # setup logger
    logger = setup_logger(output=filedir)
    logger.info(f"Using config file '{args.config_f}'.")
    logger.info(f"Saving experiment files to '{filedir}'.")

    # setup wandb
    if args.log_wandb:
        logger.info("Logging to wandb active.")
        wandb_run = wandb.init(
            project=cfg.PROJECT.CODE_DIR,
            group=cfg.MODEL.ID,
            job_type=cfg.QUANTILE_REGRESSION.FREEZE_PARAMS,
        )
    else:
        logger.info("Logging to wandb inactive.")
        wandb_run = None

    # set seed & device
    util.set_seed(cfg.PROJECT.SEED, logger=logger)
    cfg, _ = util.set_device(cfg, args.device, logger=logger)

    # create model config
    cfg_model = get_cfg()
    cfg_model.merge_from_file(model_zoo.get_config_file(cfg.MODEL.FILE))
    cfg_model.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.MODEL.FILE)
    cfg_model.merge_from_other_cfg(cfg.MODEL.CONFIG)
    cfg_model.OUTPUT_DIR = filedir
    cfg_model.freeze()

    default_setup(cfg_model, args)  # seems ok to call this

    return cfg, cfg_model, filedir, logger, wandb_run


def main(args):
    # cfg_all is the full config file, cfg is the d2 model config file
    cfg_all, cfg, filedir, logger, wandb_run = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg_all, cfg, filedir, wandb_run, logger)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def create_parser():
    parser = argparse.ArgumentParser(
        epilog="+++ Training script for Trainer +++",
        description="Parser for CLI arguments to train model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_f",  # using config_file causes reading issue with default_setup()
        type=str,
        default=None,
        required=True,
        help="Config file name to get settings to use for current run.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="conformalbb/config",
        required=False,
        help="Path to config file to use for current run.",
    )
    parser.add_argument(
        "--log_wandb",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If log run to wandb (bool).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False,
        help="Device to run code on (cpu, cuda).",
    )
    parser.add_argument(
        "--file_name_prefix",
        type=str,
        default=None,
        required=False,
        help="File name prefix to save/load results under.",
    )
    parser.add_argument(
        "--file_name_suffix",
        type=str,
        default="",
        required=False,
        help="File name suffix to save/load results under.",
    )
    # see engine/defaults.default_argument_parser() for below args
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument(
        "--eval-only",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="perform evaluation only",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="number of gpus *per machine*",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        required=False,
        default=1,
        help="total number of machines",
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        required=False,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
