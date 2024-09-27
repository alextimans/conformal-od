import sys
sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")

import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from detectron2.data import MetadataCatalog, get_detection_dataset_dicts
from detectron2.structures import Boxes, Instances
from detectron2.utils.logger import setup_logger

from model import model_loader, ensemble_boxes_wbf
from data import data_loader
from data.data_collector import DataCollector, _default_dict_fields
from calibration import random_split, pred_intervals
from evaluation import metrics, results_table
from util import util, io_file
from util.util import set_seed


class DeepEns:
    """
    Deep ensemble baseline, generating prediction intervals on the basis
    of ensemble predictions and uncertainties and a uni-modal Gaussian assumption
    on box coordinates. Comes without explicit finite-sample guarantees on coverage.
    The same Gaussian assumption is used irrespective of underlying class, and there is no class label set generation
    since there is no explicit finite-sample guarantee on coverage that needs to be controlled.
    
    NOTE: Ensemble box predictions are assumed to exist from a previous run (e.g. via EnsConformal)
    and are loaded from a specified file. Thus, only the risk control procedure is run here.    
    CLI run example:
    python conformalbb/control/baseline_deep_ens.py --config_file=cfg_base_deep_ens --config_path=conformalbb/config/coco_val --load_collect_pred=ens_conf_x101fpn_base_ens_class --risk_control=deep_ens --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --file_name_suffix=_base_deep_ens --device=cuda

    Ref: https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
    """

    def __init__(self, cfg, args, nr_class, filedir, log, logger):
        self.seed = cfg.PROJECT.SEED
        self.nr_class = nr_class
        self.filedir = filedir
        self.log = log
        self.logger = logger

        self.device = cfg.MODEL.DEVICE
        self.nr_metrics = 12

        self.calib_fraction = cfg.CALIBRATION.FRACTION
        self.calib_trials = cfg.CALIBRATION.TRIALS
        self.calib_one_sided = cfg.CALIBRATION.ONE_SIDED_PI
        self.calib_alpha = args.alpha

        self.cfg = cfg
        self.ens_size = cfg.MODEL.ENSEMBLE.SIZE

    def set_collector(self, nr_class: int, nr_img: int, dict_fields: list = []):
        self.collector = DeepEnsDataCollector(
            nr_class,
            nr_img,
            dict_fields,
            self.calib_one_sided,
            self.logger,
        )

    def __call__(self, img_list: list, ist_list: list):
        """
        Runs the risk control procedure for a set nr of trials and records
        all relevant information per trial and class.
        Here: deep ensemble baseline.

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
                # Not required for deep ensemble baseline, but we evaluate on
                # test data only, across multiple trials, for consistent comparison.
                calib_mask, _, test_idx = random_split.random_split(
                    imgs, ist["img_id"], self.calib_fraction
                )

                # Fill local tensors used to compute quantities
                scores = torch.zeros((nr_ist, self.collector.nr_scores))
                gt, pred, unc = (
                    torch.zeros_like(scores),
                    torch.zeros_like(scores),
                    torch.zeros_like(scores),
                )
                for i, s in enumerate(self.collector.score_fields):
                    scores[:, i] = ist[s]
                    # Fill with same 4 tensors (per coord) every mod 4
                    gt[:, i] = ist["gt_" + self.collector.coord_fields[i % 4]]
                    pred[:, i] = ist["pred_" + self.collector.coord_fields[i % 4]]
                    unc[:, i] = ist["unc_" + self.collector.coord_fields[i % 4]]

                if scores[calib_mask].shape[0] == 0:  # degenerate case
                    continue

                # N(0, 1) quantiles at target alpha
                nom_ql, nom_ql = self.calib_alpha / 2, (1 - self.calib_alpha / 2)
                ql, qh = torch.distributions.Normal(0, 1).icdf(
                    torch.tensor([nom_ql, nom_ql])
                )
                assert abs(ql) == abs(qh), "N(0, 1) quantiles not symmetric"
                quant = torch.full((scores.shape[1],), qh, dtype=torch.float32)

                # Compute other relevant quantities coordinate/score-wise
                nr_calib_samp = calib_mask.sum().repeat(self.collector.nr_scores)

                if self.calib_one_sided:
                    pi = pred_intervals.one_sided_pi(
                        pred, unc, quant, ist["pred_centers"]
                    )
                else:
                    pi = pred_intervals.norm_pi(pred, unc, quant)

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

                del scores, gt, pred, pi, calib_mask, test_idx

        # run label set loop (not applicable here)
        label_sets, label_data, box_set_data = None, None, None

        return data, test_indices, label_sets, label_data, box_set_data

    def evaluate(
        self,
        data: torch.Tensor,
        label_data,
        box_set_data,
        metadata: dict,
        filedir: str,
        save_file: bool,
        load_collect_pred,
    ):
        self.logger.info("Collecting and computing results...")
        assert (
            label_data is None and box_set_data is None and load_collect_pred is None
        ), "DeepEns does not support label sets."

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


class DeepEnsDataCollector(DataCollector):
    def __init__(
        self,
        nr_class: int,
        nr_img: int,
        dict_fields: list = [],
        one_sided: bool = False,
        logger=None,
        label_set_generator=None,
    ):
        if not dict_fields:
            dict_fields = _default_dict_fields.copy()
            self.coord_fields = ["x0", "y0", "x1", "y1"]
            # Uncertainty scores
            self.unc_fields = ["unc_x0", "unc_y0", "unc_x1", "unc_y1"]
            # Conformal scores
            self.score_fields = [
                "norm_res_x0",
                "norm_res_y0",
                "norm_res_x1",
                "norm_res_y1",
            ]
            self.nr_scores = len(self.score_fields)
            dict_fields += self.score_fields
            dict_fields += self.unc_fields
        super().__init__(nr_class, nr_img, dict_fields, logger, label_set_generator)
        self.one_sided = one_sided


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
        default="deep_ens",
        required=True,
        choices=["deep_ens"],
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
    data_list = get_detection_dataset_dicts(
        data_name, filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY
    )
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])

    nr_img = len(data_list)
    del data_list

    # Initialize risk control object (controller)
    logger.info(f"Init risk control procedure with '{args.risk_control}'...")
    if args.risk_control == "deep_ens":
        controller = DeepEns(cfg, args, nr_class, filedir, log=False, logger=logger)
    else:
        raise ValueError("Risk control procedure not specified.")

    # Initialize relevant DataCollector object
    controller.set_collector(nr_class, nr_img)  # type: ignore

    # Get prediction information & risk control scores
    assert (
        args.load_collect_pred is not None
    ), "Need to load predictions to run baseline."
    pred_filedir = os.path.join(outdir, data_name, args.load_collect_pred)
    logger.info(f"Loading existing predictions from '{pred_filedir}'.")
    img_list = io_file.load_json(f"{args.load_collect_pred}_img_list", pred_filedir)
    ist_list = io_file.load_json(f"{args.load_collect_pred}_ist_list", pred_filedir)

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
