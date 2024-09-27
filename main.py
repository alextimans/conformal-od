import sys
sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")

import os
import argparse
import wandb
from pathlib import Path

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts

from util import util, io_file
from data import data_loader
from model import model_loader
from control import std_conformal, ens_conformal, cqr_conformal, baseline_conformal
from model.qr_head import QuantileROIHead


def create_parser():
    """
    This function creates a parser for CLI arguments to initialize
    the model and experiment settings.
    Run 'python main.py -h' to see the help messages for each argument.
    The args hierarchy: CLI > cfg > cfg_model default > d2_model default
    """
    parser = argparse.ArgumentParser(
        description="Parser for CLI arguments to run model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_file",
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
        "--run_collect_pred",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If run collect_predictions method (bool).",
    )
    parser.add_argument(
        "--load_collect_pred",
        type=str,
        default=None,
        required=False,
        help="File name prefix from which to load pred info if not running collect_predictions",
    )
    parser.add_argument(
        "--save_file_pred",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If save collect_predictions results to file (bool).",
    )
    parser.add_argument(
        "--risk_control",
        type=str,
        default=None,
        required=True,
        choices=["std_conf", "ens_conf", "cqr_conf", "base_conf"],
        help="Type of risk control/conformal approach to use.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        required=False,
        help="Alpha level for box coverage guarantee.",
    )
    parser.add_argument(
        "--label_set",
        type=str,
        default="top_singleton",
        required=False,
        choices=["top_singleton", "full", "oracle", "class_threshold"],
        help="Type of label set construction to use.",
    )
    parser.add_argument(
        "--label_alpha",
        type=float,
        default=0.01,
        required=False,
        help="Alpha level for label set coverage guarantee.",
    )
    parser.add_argument(
        "--run_risk_control",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If run risk control procedure, i.e. controller.__call__ (bool).",
    )
    parser.add_argument(
        "--load_risk_control",
        type=str,
        default=None,
        required=False,
        help="File name prefix from which to load control info if not running risk control",
    )
    parser.add_argument(
        "--save_file_control",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If save risk control procedure results to file (bool).",
    )
    parser.add_argument(
        "--save_label_set",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If save predicted label sets to file (bool).",
    )
    parser.add_argument(
        "--run_eval",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If run risk control evaluation, i.e. controller.evaluate (bool).",
    )
    parser.add_argument(
        "--save_file_eval",
        action=argparse.BooleanOptionalAction,
        required=False,
        help="If save results table to file (bool).",
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
    return parser


def main():
    """
    This function is the main function to run the model and experiment.
    It includes the following steps:
    1. Parse CLI arguments and load config file.
    2. Set up logging and seed.
    3. Register data with detectron2.
    4. Build model and load model checkpoint.
    5. Load data for full dataset.
    6. Initialize risk control object (controller) and DataCollector object.
    7. Get prediction information & risk control scores.
    8. Get risk control procedure output.
    9. Get results tables.
    
    Note that for the object detector baselines (DeepEns, GaussianYOLOv3, YOLOv3, DETR, Sparse R-CNN)
    we have self-contained mains in each respective conformal python script since 
    the model loading and data loading is different from the detectron2 models.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Load config
    cfg = io_file.load_yaml(args.config_file, args.config_path, to_yacs=True)
    data_name = cfg.DATASETS.DATASET.NAME

    # Determine file naming and create experiment folder
    if args.file_name_prefix is not None:
        file_name_prefix = args.file_name_prefix
    else:
        file_name_prefix = (
            f"{args.risk_control}_{cfg.MODEL.ID}{args.file_name_suffix}"
        )
    outdir = cfg.PROJECT.OUTPUT_DIR
    filedir = os.path.join(outdir, data_name, file_name_prefix)
    Path(filedir).mkdir(exist_ok=True, parents=True)

    # Set up logging
    logger = setup_logger(output=filedir)
    logger.info("Running %s..." % (sys._getframe().f_code.co_name))
    logger.info(f"Using config file '{args.config_file}'.")
    logger.info(f"Saving experiment files to '{filedir}'.")

    # Set seed & device
    util.set_seed(cfg.PROJECT.SEED, logger=logger)
    cfg, _ = util.set_device(cfg, args.device, logger=logger)

    # Register data with detectron2
    data_loader.d2_register_dataset(cfg, logger=logger)
    # Build model and load model checkpoint
    cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
    model_loader.d2_load_model(cfg_model, model, logger=logger)
    # Load data for full dataset
    data_list = get_detection_dataset_dicts(
        data_name, filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY
    )
    dataloader = data_loader.d2_load_dataset_from_dict(
        data_list, cfg, cfg_model, logger=logger
    )
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])

    # Log to wandb (optional)
    if args.log_wandb:
        logger.info("Logging to wandb active.")
        wandb_run = wandb.init(
            project=cfg.PROJECT.CODE_DIR,
            group=cfg.DATASETS.DATASET.NAME,
            job_type=args.risk_control,
        )
    else:
        logger.info("Logging to wandb inactive.")
        wandb_run = None

    # Initialize risk control object (controller)
    logger.info(f"Init risk control procedure with '{args.risk_control}'...")
    if args.risk_control == "std_conf":
        controller = std_conformal.StdConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    elif args.risk_control == "ens_conf":
        controller = ens_conformal.EnsConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    elif args.risk_control == "cqr_conf":
        controller = cqr_conformal.CQRConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    elif args.risk_control == "base_conf":
        controller = baseline_conformal.BaselineConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    else:
        raise ValueError("Risk control procedure not specified.")

    # Initialize relevant DataCollector object
    controller.set_collector(nr_class, len(data_list))  # type: ignore

    # Get prediction information & risk control scores
    if args.run_collect_pred:
        logger.info("Collecting predictions...")
        img_list, ist_list = controller.collect_predictions(model, dataloader)
        if args.save_file_pred:
            controller.collector.to_file(file_name_prefix, filedir)
    elif args.load_collect_pred is not None:
        pred_filedir = os.path.join(outdir, data_name, args.load_collect_pred)
        logger.info(f"Loading existing predictions from '{pred_filedir}'.")
        img_list = io_file.load_json(f"{args.load_collect_pred}_img_list", pred_filedir)
        ist_list = io_file.load_json(f"{args.load_collect_pred}_ist_list", pred_filedir)
    else:
        logger.info(f"Loading existing predictions from '{filedir}'.")
        img_list = io_file.load_json(f"{file_name_prefix}_img_list", filedir)
        ist_list = io_file.load_json(f"{file_name_prefix}_ist_list", filedir)

    # Get risk control procedure output
    if args.run_risk_control:
        logger.info("Running risk control procedure...")
        control_data, test_indices, label_sets, label_data, box_set_data = controller(
            img_list, ist_list
        )
        if args.save_file_control:
            io_file.save_tensor(control_data, f"{file_name_prefix}_control", filedir)
            io_file.save_tensor(test_indices, f"{file_name_prefix}_test_idx", filedir)
            io_file.save_tensor(label_data, f"{file_name_prefix}_label", filedir)
            io_file.save_tensor(box_set_data, f"{file_name_prefix}_box_set", filedir)
        if args.save_label_set:
            io_file.save_tensor(label_sets, f"{file_name_prefix}_label_set", filedir)
    elif args.load_risk_control is not None:
        control_filedir = os.path.join(outdir, data_name, args.load_risk_control)
        logger.info(f"Loading existing control files from '{control_filedir}'.")
        control_data = io_file.load_tensor(
            f"{args.load_risk_control}_control", control_filedir
        )
        test_indices = io_file.load_tensor(
            f"{args.load_risk_control}_test_idx", control_filedir
        )
        label_data = io_file.load_tensor(
            f"{args.load_risk_control}_label", control_filedir
        )
        box_set_data = io_file.load_tensor(
            f"{args.load_risk_control}_box_set", control_filedir
        )
    else:
        logger.info(f"Loading existing control files from '{filedir}'.")
        control_data = io_file.load_tensor(f"{file_name_prefix}_control", filedir)
        test_indices = io_file.load_tensor(f"{file_name_prefix}_test_idx", filedir)
        label_data = io_file.load_tensor(f"{file_name_prefix}_label", filedir)
        box_set_data = io_file.load_tensor(f"{file_name_prefix}_box_set", filedir)

    # Get results tables
    if args.run_eval:
        logger.info("Evaluating risk control...")
        controller.evaluate(
            control_data,
            label_data,
            box_set_data,
            metadata,
            filedir,
            args.save_file_eval,
            args.load_collect_pred,
        )


if __name__ == "__main__":
    main()
