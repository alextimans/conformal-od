import os
import torch
from tqdm import tqdm

from detectron2.structures.boxes import Boxes
from detectron2.data.detection_utils import annotations_to_instances

from .abstract_risk_control import RiskControl
from model import matching
from data.data_collector import DataCollector, _default_dict_fields
from calibration import random_split, pred_intervals
from calibration.conformal_scores import one_sided_res, one_sided_mult_res
from evaluation import metrics, results_table, ap_evaluation
from util.util import set_seed
from control.classifier_sets import get_label_set_generator


class BaselineConformal(RiskControl):
    """
    This class is used to run the conformal bounding box procedures with
    nonconformity scores from our conformal baselines.
    
    Specifically, we use the one-sided residual and one-sided multiplicative
    residual scores from DeGrancey et al (2022) and Andeol et al (2023),
    and also adapt them to the two-sided case.
    
    The preceeding class label set step is integrated by the label_set_generator
    attribute, which is a callable object that generates label sets and computes
    the relevant metrics, and is called in the __call__ method.
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
        self.calib_alpha = args.alpha

        self.ap_eval = cfg.MODEL.AP_EVAL
        if self.ap_eval:
            self.ap_evaluator = ap_evaluation.APEvaluator(nr_class=nr_class)

        self.label_alpha = args.label_alpha
        self.label_set_generator = get_label_set_generator(cfg, args, logger)

    def set_collector(self, nr_class: int, nr_img: int, dict_fields: list = []):
        """
        Initializes the data collector for the risk control procedure.
        """
        self.collector = BaselineConformalDataCollector(
            nr_class, nr_img, dict_fields, self.logger, self.label_set_generator
        )

    def raw_prediction(self, model, img):
        """
        Runs the model on a single image and returns the raw prediction.
        """        
        model.eval()
        with torch.no_grad():
            pred = model([img])
        return pred[0]["instances"]

    def collect_predictions(self, model, dataloader, verbose: bool = False):
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
        with torch.no_grad(), tqdm(dataloader, desc="Images") as loader:
            for i, img in enumerate(loader):
                # BoxMode.XYWH to BoxMode.XYXY and correct formatting
                gt = annotations_to_instances(
                    img[0]["annotations"], (img[0]["height"], img[0]["width"])
                )
                gt_box, gt_class = gt.gt_boxes, gt.gt_classes

                pred = model(img)
                pred_ist = pred[0]["instances"].to("cpu")
                pred_box = pred_ist.pred_boxes
                pred_class = pred_ist.pred_classes
                pred_score = pred_ist.scores
                pred_score_all = pred_ist.scores_all

                # Collect and store AP eval info
                if self.ap_eval:
                    self.ap_evaluator.collect(
                        gt_box, pred_box, gt_class, pred_class, pred_score
                    )

                # Object matching process (predictions to ground truths)
                (
                    gt_box,
                    pred_box,
                    gt_class,
                    pred_class,
                    pred_score,
                    pred_score_all,
                    _,  # pred_logits_all
                    matches,
                ) = matching.matching(
                    gt_box,
                    pred_box,
                    gt_class,
                    pred_class,
                    pred_score,
                    pred_score_all,
                    box_matching=self.box_matching,
                    class_match=self.class_matching,
                    thresh=self.iou_thresh,
                )
                # Collect and store risk control info
                if matches:
                    self.collector(
                        gt_box, gt_class, pred_box, pred_score, pred_score_all, img_id=i
                    )

                if self.log is not None:
                    self.log.define_metric("nr_matches", summary="mean")
                    self.log.log({"nr_matches": len(gt_class)})

                if verbose:
                    self.logger.info(f"\n{gt_class=}\n{pred_class=},\n{pred_score=}")

                del gt, pred, pred_ist

        if self.ap_eval:
            self.ap_evaluator.to_file(
                self.ap_evaluator.ap_info, "ap_info", self.filedir
            )
        return self.collector.img_list, self.collector.ist_list

    def __call__(self, img_list: list, ist_list: list):
        """
        Runs the risk control procedure for a set nr of trials and records
        all relevant information per trial and class.

        Args:
            img_list (list): See DataCollector docstring
            ist_list (list): See DataCollector docstring

        Returns:
            data (Tensor): Tensor containing coordinate/score-wise
            information such as quantile or coverage per trial and class.
            test_indices (Tensor): Boolean tensor recording which
            images end up in the test set (vs. calibration set) per trial
            and class.
        """
        self.logger.info(
            f"""
            Running risk control procedure for {self.calib_trials} trials...
            Calibration fraction: {self.calib_fraction}, alpha: {self.calib_alpha},
            box correction: {self.calib_box_corr}.
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

        # Collect label set information
        self.label_set_generator.collect(
            img_list,
            ist_list,
            self.nr_class,
            self.nr_metrics,
            self.collector.nr_scores,
            self.collector.score_fields,
            self.collector.coord_fields,
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
                    # Fill with same tensors mod 4 (per coord)
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

                # Compute label set quantiles and store info
                self.label_set_generator.calib_masks[t][c] = calib_mask
                self.label_set_generator.box_quantiles[t, c] = quant
                self.label_set_generator.label_quantiles[
                    t, c
                ] = pred_intervals.get_quantile(
                    scores=ist["label_score"][calib_mask],
                    alpha=self.label_alpha,
                    n=calib_mask.sum(),
                )

                # Compute other relevant quantities coordinate/score-wise
                nr_calib_samp = calib_mask.sum().repeat(self.collector.nr_scores)

                # Scaling factors for one sided PIs
                # For additive scores, these are just tensors of ones
                scale_add = torch.ones((pred.shape[0], 4))
                # For multiplicative scores, these are (width, height, width, height)
                scale_mult = torch.stack(
                    (
                        (pred[:, 2] - pred[:, 0]),
                        (pred[:, 3] - pred[:, 1]),
                        (pred[:, 2] - pred[:, 0]),
                        (pred[:, 3] - pred[:, 1]),
                    ),
                    dim=1,
                )
                # Approach adapted to create two-sided PIs by considering box centers as the lower bounds
                pi = torch.cat(
                    (
                        pred_intervals.one_sided_pi(  # for one_sided_res
                            pred[:, :4], scale_add, quant[:4], ist["pred_centers"]
                        ),
                        pred_intervals.one_sided_pi(  # for one_sided_mult_res
                            pred[:, 4:], scale_mult, quant[4:], ist["pred_centers"]
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

        # run label set loop, which also returns box_set_data containing results with label sets
        label_sets, label_data, box_set_data = self.label_set_generator()

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

        # AP eval
        if self.ap_eval:
            self.ap_evaluator.evaluate(
                metadata["thing_classes"],
                filedir,
                load_collect_pred,
                logger=self.logger,
            )

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

            results_table.get_label_results_table(
                data=label_data,
                class_names=metadata["thing_classes"],
                to_file=save_file,
                filename=f"{os.path.split(filedir)[-1]}_label_table",
                filedir=filedir,
                logger=self.logger,
            )

            results_table.get_box_set_results_table(
                data=box_set_data[:, :, i : (i + 4), :],
                class_names=metadata["thing_classes"],
                to_file=save_file,
                filename=f"{os.path.split(filedir)[-1]}_box_set_table_{self.collector.score_fields[i][:-3]}",
                filedir=filedir,
                logger=self.logger,
            )


class BaselineConformalDataCollector(DataCollector):
    """
    Subclass of DataCollector for the BaselineConformal risk control procedure.
    The stored information is customized to each conformal procedure, e.g.,
    on storing of the conformal scores.
    """
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
                "one_sided_res_x0",
                "one_sided_res_y0",
                "one_sided_res_x1",
                "one_sided_res_y1",
                "one_sided_mult_res_x0",
                "one_sided_mult_res_y0",
                "one_sided_mult_res_x1",
                "one_sided_mult_res_y1",
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
        pred_score_all: torch.Tensor,
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

            width = pred_box[idx].tensor[:, 2] - pred_box[idx].tensor[:, 0]  # x1-x0
            height = pred_box[idx].tensor[:, 3] - pred_box[idx].tensor[:, 1]  # y1-y0

            self.ist_list[c]["one_sided_mult_res_x0"] += one_sided_mult_res(
                gt_box[idx].tensor[:, 0],
                pred_box[idx].tensor[:, 0],
                mult=width,
                min=True,
            ).tolist()
            self.ist_list[c]["one_sided_mult_res_y0"] += one_sided_mult_res(
                gt_box[idx].tensor[:, 1],
                pred_box[idx].tensor[:, 1],
                mult=height,
                min=True,
            ).tolist()
            self.ist_list[c]["one_sided_mult_res_x1"] += one_sided_mult_res(
                gt_box[idx].tensor[:, 2],
                pred_box[idx].tensor[:, 2],
                mult=width,
                min=False,
            ).tolist()
            self.ist_list[c]["one_sided_mult_res_y1"] += one_sided_mult_res(
                gt_box[idx].tensor[:, 3],
                pred_box[idx].tensor[:, 3],
                mult=height,
                min=False,
            ).tolist()

        if verbose:
            print(f"Added all instances for image {img_id}.")
