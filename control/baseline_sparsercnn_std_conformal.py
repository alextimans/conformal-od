import sys
sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")

import os
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.data.detection_utils import annotations_to_instances

from abstract_risk_control import RiskControl
from data import data_loader
from model import model_loader, matching
from data.data_collector import DataCollector, _default_dict_fields
from calibration import random_split, pred_intervals
from calibration.conformal_scores import abs_res, one_sided_res
from evaluation import metrics, results_table
from util import util, io_file
from util.util import set_seed


class SparseRCNNStdConformal(RiskControl):
    """
    Model baseline: Sparse R-CNN + StdConformal (as opposed to using Faster R-CNN from detectron2).
    Code reuses a lot from std_conformal.py and adds model-specific parts.
    
    Model from: https://github.com/PeizeSun/SparseR-CNN/
    
    CLI run examples:
    python conformalbb/control/baseline_sparsercnn_std_conformal.py --config_file=cfg_base_sparsercnn --config_path=conformalbb/config/coco_val --run_collect_pred --save_file_pred --risk_control=std_conf --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_base_std_rank --device=cuda
    python conformalbb/control/baseline_sparsercnn_std_conformal.py --config_file=cfg_base_sparsercnn --config_path=conformalbb/config/coco_val --load_collect_pred=std_conf_sparsercnn_base_std_rank --risk_control=std_conf --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_base_std_max --device=cuda
    
    NOTE: Requires a generated prediction file from Sparse R-CNN to run, which contains
    the model's box proposals for each image in the dataset (COCO).
    To get the proposals:
    - Follow "https://github.com/PeizeSun/SparseR-CNN/" for eval mode
    - Download and place model checkpoint in checkpoint folder (model R101-300pro)
    - Run cmd: python projects/SparseRCNN/train_net.py --num-gpus 1 --config-file projects/SparseRCNN/configs/sparsercnn.res101.300pro.3x.yaml --eval-only MODEL.WEIGHTS ../checkpoints/sparse_rcnn_r101_300pro.pth
    - This will generate a file "SparseR-CNN/output/inference/instances_predictions.pth" with 300 box proposals per COCO image which can be used for this script.
    """
    
    def __init__(self, cfg, args, nr_class, filedir, log, logger):
        self.seed = cfg.PROJECT.SEED
        self.nr_class = nr_class
        self.filedir = filedir
        self.log = log
        self.logger = logger

        self.device = cfg.MODEL.DEVICE
        self.box_matching = cfg.MODEL.BOX_MATCHING
        self.class_matching = cfg.MODEL.CLASS_MATCHING
        self.iou_thresh = cfg.MODEL.IOU_THRESH_TEST
        self.nr_metrics = 12

        self.calib_fraction = cfg.CALIBRATION.FRACTION
        self.calib_trials = cfg.CALIBRATION.TRIALS
        self.calib_box_corr = cfg.CALIBRATION.BOX_CORRECTION
        self.calib_one_sided = cfg.CALIBRATION.ONE_SIDED_PI
        self.calib_alpha = args.alpha

        # sparse rcnn-specific
        self.conf_thresh = cfg.MODEL.CONFIG.MODEL.ROI_HEADS.SCORE_THRESH_TEST

    def set_collector(self, nr_class: int, nr_img: int, dict_fields: list = []):
        self.collector = SparseRCNNStdConformalDataCollector(
            nr_class,
            nr_img,
            dict_fields,
            self.logger,
        )

    def collect_predictions(
        self, model, dataloader, proposals, verbose: bool = False
    ):
        """
        Cycles through the dataloader containing the full eval dataset
        and collects information on model predictions and matched
        ground truths relevant for downstream risk control.

        Args:
            model (torch.nn.Module): Loaded model
            dataloader (torch.data.DataLoader): Dataloader for full dataset
            verbose (bool, optional): Defaults to False.

        Returns:
            img_list (list), ist_list (list)
        """
        self.logger.info(
            f"""
            Running 'collect_predictions' with {self.iou_thresh=}...
            Box matching: '{self.box_matching}', class matching: {self.class_matching}.
            """
        )

        model.eval()
        model.to(self.device)

        with torch.no_grad(), tqdm(dataloader, desc="Images") as loader:
            for i, img in enumerate(loader):
                
                # BoxMode.XYWH to BoxMode.XYXY and correct formatting
                gt = annotations_to_instances(
                    img[0]["annotations"], (img[0]["height"], img[0]["width"])
                )
                gt_box, gt_class = gt.gt_boxes, gt.gt_classes
                if gt_box is None:  # degenerate case
                    continue
                
                # Load Sparse R-CNN proposals (in COCO format)
                preds = proposals[i]["instances"]
                
                scores = torch.tensor([el["score"] for el in preds])
                cl = torch.tensor([el["category_id"] for el in preds])
                boxes = torch.tensor([BoxMode.convert(el["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for el in preds])
                
                # keep only preds above conf_thresh if desired
                # mask = (scores >= self.conf_thresh)
                mask = torch.ones(scores.shape[0], dtype=torch.bool)
                
                # Get pred into right format (detectron2 style)
                pred_class = cl[mask]
                pred_score = scores[mask]
                pred_box = Boxes(boxes[mask])

                # Matching process
                (
                    gt_box,
                    pred_box,
                    gt_class,
                    pred_class,
                    pred_score,
                    _,
                    _,
                    matches,
                    _,
                    pred_idx,
                    class_idx,
                ) = matching.matching(
                    gt_box,
                    pred_box,
                    gt_class,
                    pred_class,
                    pred_score,
                    box_matching=self.box_matching,
                    class_match=self.class_matching,
                    thresh=self.iou_thresh,
                    return_idx=True,
                )
                
                # Collect and store risk control info
                if matches:
                    self.collector(
                        gt_box,
                        gt_class,
                        pred_box,
                        # pred_unc,
                        pred_score,
                        img_id=i,
                    )

                del img, preds

        return self.collector.img_list, self.collector.ist_list

    def __call__(self, img_list: list, ist_list: list):
        """
        Runs the risk control procedure for a set nr of trials and records
        all relevant information per trial and class.
        Here: standard split conformal.

        Args:
            img_list (list): See DataCollector docstring
            ist_list (list): See DataCollector docstring

        Returns:
            data (torch.tensor): Tensor containing coordinate/score-wise
            information such as quantile or coverage per trial and class.
            test_indices (torch.tensor): Boolean tensor recording which
            images end up in the test set (vs. calibration set) per trial
            and class.
        """
        self.logger.info(
            f"""
            Running risk control procedure for {self.calib_trials} trials...
            Calibration fraction: {self.calib_fraction}, alpha: {self.calib_alpha}.
            """
        )

        # Tensors to store information in
        data = torch.zeros(
            size=(
                self.calib_trials,
                self.nr_class,
                self.collector.nr_scores,
                self.nr_metrics,
            ),
            dtype=torch.float32,
        )
        test_indices = torch.zeros(
            size=(self.calib_trials, self.nr_class, self.collector.nr_img),
            dtype=torch.bool,
        )

        for t in tqdm(range(self.calib_trials), desc="Trials"):
            set_seed((self.seed + t), self.logger, False)  # New seed for each trial

            for c in range(self.nr_class):
                imgs = torch.tensor(img_list[c])
                ist = {k: torch.tensor(v) for k, v in ist_list[c].items()}
                nr_ist = len(ist["img_id"])

                if nr_ist == 0:  # no ist for this class
                    continue

                # Get calibration/test split
                calib_mask, _, test_idx = random_split.random_split(
                    imgs, ist["img_id"], self.calib_fraction
                )

                # Fill local tensors used to compute quantities
                scores = torch.zeros((nr_ist, self.collector.nr_scores))
                gt, pred = torch.zeros_like(scores), torch.zeros_like(scores)
                for i, s in enumerate(self.collector.score_fields):
                    scores[:, i] = ist[s]
                    # Fill with same 4 tensors (per coord) every mod 4
                    gt[:, i] = ist["gt_" + self.collector.coord_fields[i % 4]]
                    pred[:, i] = ist["pred_" + self.collector.coord_fields[i % 4]]

                if scores[calib_mask].shape[0] == 0:  # degenerate case
                    continue

                # Compute quantiles coordinate-wise incl. box correction scheme
                quant = pred_intervals.compute_quantile(
                    scores=scores[calib_mask],
                    box_correction=self.calib_box_corr,
                    alpha=self.calib_alpha,
                )

                # Compute other relevant quantities coordinate/score-wise
                nr_calib_samp = calib_mask.sum().repeat(self.collector.nr_scores)
                
                # Scaling factors for one sided PIs
                # For additive scores, these are just tensors of ones
                scale_add = torch.ones((pred.shape[0], 4))
                
                pi = torch.cat(
                    (
                        pred_intervals.fixed_pi(
                            pred[:, :4], quant[:4]
                        ),
                        pred_intervals.one_sided_pi(
                            pred[:, 4:], scale_add, quant[4:], ist["pred_centers"]
                        ),
                    ),
                    dim=-1,
                )
                
                cov_coord, cov_box = metrics.coverage(gt[~calib_mask], pi[~calib_mask])
                cov_area, cov_iou = metrics.stratified_coverage(gt, pi, calib_mask, ist)
                mpiw = metrics.mean_pi_width(pi[~calib_mask])
                stretch = metrics.box_stretch(
                    pi[~calib_mask], ist["pred_area"][~calib_mask]
                )

                # Store information
                # NOTE: order of metrics fed into tensor matters for results tables and plotting
                metr = torch.stack(
                    (nr_calib_samp, quant, mpiw, stretch, cov_coord, cov_box), dim=1
                )
                metr = torch.cat((metr, cov_area, cov_iou), dim=1)

                data[t, c, :, :] = metr
                test_indices[t, c, test_idx.sort()[0]] = True

                # if self.log is not None:
                #     log something to wandb

                del scores, gt, pred, pi, calib_mask, test_idx

        # run label set loop (not applicable here)
        label_sets, label_data, box_set_data = None, None, None

        return data, test_indices, label_sets, label_data, box_set_data

    def evaluate(
        self,
        data: torch.Tensor,
        label_data: torch.Tensor,
        box_set_data: torch.Tensor,
        metadata: dict,
        filedir: str,
        save_file: bool,
        load_collect_pred,
    ):
        self.logger.info("Collecting and computing results...")
        assert (
            label_data is None and box_set_data is None and load_collect_pred is None
        ), "SparseRCNNStdConformal is currently not evaluted for label sets."

        # Risk control eval
        for s in range(self.collector.nr_scores // 4):
            i = s * 4
            self.logger.info(
                f"Evaluating for scores {self.collector.score_fields[i:(i+4)]}"
            )

            results_table.get_results_table(
                data=data[:, :, i : (i + 4), :],
                class_names=metadata["thing_classes"],
                to_file=save_file,
                filename=f"{os.path.split(filedir)[-1]}_res_table_{self.collector.score_fields[i][:-3]}",
                filedir=filedir,
                logger=self.logger,
            )


class SparseRCNNStdConformalDataCollector(DataCollector):
    def __init__(
        self,
        nr_class: int,
        nr_img: int,
        dict_fields: list = [],
        logger=None,
        label_set_generator=None,
    ):
        if not dict_fields:
            dict_fields = _default_dict_fields.copy()
            self.coord_fields = ["x0", "y0", "x1", "y1"]
            # Conformal scores
            self.score_fields = [
                "abs_res_x0",
                "abs_res_y0",
                "abs_res_x1",
                "abs_res_y1",
                "one_sided_res_x0",
                "one_sided_res_y0",
                "one_sided_res_x1",
                "one_sided_res_y1",
            ]
            self.nr_scores = len(self.score_fields)
            dict_fields += self.score_fields
        super().__init__(nr_class, nr_img, dict_fields, logger, label_set_generator)

    def __call__(
        self,
        gt_box: Boxes,
        gt_class: torch.Tensor,
        pred_box: Boxes,
        pred_score: torch.Tensor,
        pred_score_all: torch.Tensor = None,
        pred_logits_all: torch.Tensor = None,
        img_id: int = None,
        verbose: bool = False,
    ):
        for c in torch.unique(gt_class).numpy():
            # img has instances of class
            self.img_list[c][img_id] = 1
            # indices for matching instances
            idx = torch.nonzero(gt_class == c, as_tuple=True)[0]
            # Add base infos
            super()._add_instances(
                c,
                img_id,
                idx,
                gt_box,
                pred_box,
                pred_score,
                pred_score_all,
                pred_logits_all,
            )

            # Add conformal scores
            self.ist_list[c]["abs_res_x0"] += abs_res(
                gt_box[idx].tensor[:, 0], pred_box[idx].tensor[:, 0]
            ).tolist()
            self.ist_list[c]["abs_res_y0"] += abs_res(
                gt_box[idx].tensor[:, 1], pred_box[idx].tensor[:, 1]
            ).tolist()
            self.ist_list[c]["abs_res_x1"] += abs_res(
                gt_box[idx].tensor[:, 2], pred_box[idx].tensor[:, 2]
            ).tolist()
            self.ist_list[c]["abs_res_y1"] += abs_res(
                gt_box[idx].tensor[:, 3], pred_box[idx].tensor[:, 3]
            ).tolist()

            # if "one_sided_res_x0" in self.dict_fields:
            self.ist_list[c]["one_sided_res_x0"] += one_sided_res(
                gt_box[idx].tensor[:, 0], pred_box[idx].tensor[:, 0], min=True
            ).tolist()
            self.ist_list[c]["one_sided_res_y0"] += one_sided_res(
                gt_box[idx].tensor[:, 1], pred_box[idx].tensor[:, 1], min=True
            ).tolist()
            self.ist_list[c]["one_sided_res_x1"] += one_sided_res(
                gt_box[idx].tensor[:, 2], pred_box[idx].tensor[:, 2], min=False
            ).tolist()
            self.ist_list[c]["one_sided_res_y1"] += one_sided_res(
                gt_box[idx].tensor[:, 3], pred_box[idx].tensor[:, 3], min=False
            ).tolist()

        if verbose:
            print(f"Added all instances for image {img_id}.")


def create_parser():
    """
    hierarchy: CLI > cfg > cfg_model default > d2_model default
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
        "--pred_file",
        type=str,
        default="SparseR-CNN/output/inference/instances_predictions.pth",
        required=False,
        help="Path to pred file to use for current run.",
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
        default="std_conf",
        required=True,
        choices=["std_conf"],
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
        "--device",
        type=str,
        default="cpu",
        required=False,
        help="Device to run code on (cpu, cuda).",
    )
    return parser


def main():
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
            f"{args.risk_control}_{cfg.MODEL.ID}{args.file_name_suffix}"  # type:ignore
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
    # Load data for full dataset
    data_list = get_detection_dataset_dicts(
        data_name, filter_empty=False  # Don't filter for Sparse R-CNN
    )
    dataloader = data_loader.d2_load_dataset_from_dict(
        data_list, cfg, cfg_model, logger=logger
    )
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])
    nr_img = len(data_list)
    
    # NOTE: See docstring
    proposals = torch.load(args.pred_file)
    
    # Initialize risk control object (controller)
    logger.info(f"Init risk control procedure with '{args.risk_control}'...")
    if args.risk_control == "std_conf":
        controller = SparseRCNNStdConformal(
            cfg, args, nr_class, filedir, log=False, logger=logger
        )
    else:
        raise ValueError("Risk control procedure not specified.")

    # Initialize relevant DataCollector object
    controller.set_collector(nr_class, nr_img)

    # Get prediction information & risk control scores
    if args.run_collect_pred:
        logger.info("Collecting predictions...")
        img_list, ist_list = controller.collect_predictions(model, dataloader, proposals)
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
        control_data, test_indices, _, _, _ = controller(img_list, ist_list)
        if args.save_file_control:
            io_file.save_tensor(control_data, f"{file_name_prefix}_control", filedir)
            io_file.save_tensor(test_indices, f"{file_name_prefix}_test_idx", filedir)
    elif args.load_risk_control is not None:
        control_filedir = os.path.join(outdir, data_name, args.load_risk_control)
        logger.info(f"Loading existing control files from '{control_filedir}'.")
        control_data = io_file.load_tensor(
            f"{args.load_risk_control}_control", control_filedir
        )
        test_indices = io_file.load_tensor(
            f"{args.load_risk_control}_test_idx", control_filedir
        )
    else:
        logger.info(f"Loading existing control files from '{filedir}'.")
        control_data = io_file.load_tensor(f"{file_name_prefix}_control", filedir)
        test_indices = io_file.load_tensor(f"{file_name_prefix}_test_idx", filedir)

    # Get results tables
    if args.run_eval:
        logger.info("Evaluating risk control...")
        controller.evaluate(
            control_data,
            None,
            None,
            metadata,
            filedir,
            args.save_file_eval,
            None,
        )


if __name__ == "__main__":
    main()
