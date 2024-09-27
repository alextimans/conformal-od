import os
import torch
from typing import List
from tqdm import tqdm
from itertools import product

from detectron2.structures.boxes import Boxes
from detectron2.data.detection_utils import annotations_to_instances

from .abstract_risk_control import RiskControl
from model import matching
from data.data_collector import DataCollector, _default_dict_fields
from calibration import random_split, pred_intervals
from calibration.conformal_scores import quant_res
from evaluation import metrics, results_table, ap_evaluation
from util.util import set_seed
from control.classifier_sets import get_label_set_generator


class CQRConformal(RiskControl):
    """
    This class is used to run the conformal bounding box procedure with
    quantile residual nonconformity scores (Box-CQR).
    We leverage the CQR procedure from Romano et al. (2019) and adapt it
    to bounding box regression using trained quantile regression heads
    on top of our pre-trained object detector.
    
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

        self.quantiles = cfg.QUANTILE_REGRESSION.QUANTILES
        self.q_str = [f"q{int(q*100)}" for q in self.quantiles]
        self.q_idx = cfg.QUANTILE_REGRESSION.QUANTILE_INDICES

        self.ap_eval = cfg.MODEL.AP_EVAL
        if self.ap_eval:
            self.ap_evaluator = ap_evaluation.APEvaluator(nr_class=nr_class)
            self.ap_evaluator_q = [
                ap_evaluation.APEvaluator(nr_class=nr_class)
                for i in range(len(self.quantiles))
            ]

        self.label_alpha = args.label_alpha
        self.label_set_generator = get_label_set_generator(cfg, args, logger)

    def set_collector(self, nr_class: int, nr_img: int, dict_fields: list = []):
        """
        Initializes the data collector for the risk control procedure.
        """
        self.collector = CQRConformalDataCollector(
            self.quantiles,
            nr_class,
            nr_img,
            dict_fields,
            self.logger,
            self.label_set_generator,
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
                pred_box_q = [pred_ist.get(f"pred_boxes_{q}") for q in self.q_str]

                # Collect and store AP eval info
                if self.ap_eval:
                    self.ap_evaluator.collect(
                        gt_box, pred_box, gt_class, pred_class, pred_score
                    )
                    for box, ap in enumerate(self.ap_evaluator_q):
                        ap.collect(
                            gt_box, pred_box_q[box], gt_class, pred_class, pred_score
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
                    _,  # gt_idx
                    pred_idx,
                    class_idx,
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
                    return_idx=True,
                )
                # Subset quantile preds by match indices
                pred_box_q = [box[pred_idx] for box in pred_box_q]
                if self.class_matching:
                    pred_box_q = [box[class_idx] for box in pred_box_q]

                # Collect and store risk control info
                if matches:
                    self.collector(
                        gt_box,
                        gt_class,
                        pred_box,
                        pred_box_q,
                        self.q_idx,
                        pred_score,
                        pred_score_all,
                        img_id=i,
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
            for box, ap in enumerate(self.ap_evaluator_q):
                ap.to_file(ap.ap_info, f"ap_info_{box}", self.filedir)

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
                gt, pred_lower, pred_upper = (
                    torch.zeros_like(scores),
                    torch.zeros_like(scores),
                    torch.zeros_like(scores),
                )
                for i, s in enumerate(self.collector.score_fields):
                    scores[:, i] = ist[s]
                    # Fill with same 4 tensors (per coord) every mod 4
                    gt[:, i] = ist["gt_" + self.collector.coord_fields[i % 4]]
                    pred_lower[:, i] = ist[  # lower: q_idx[0]
                        f"pred_{self.q_str[self.q_idx[0]]}_"
                        + self.collector.coord_fields[i % 4]
                    ]
                    pred_upper[:, i] = ist[  # upper: q_idx[1]
                        f"pred_{self.q_str[self.q_idx[1]]}_"
                        + self.collector.coord_fields[i % 4]
                    ]

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
                pi = pred_intervals.quant_pi(pred_lower, pred_upper, quant)
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

                del scores, gt, pred_lower, pred_upper, pi, calib_mask, test_idx

        # run label set loop
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
            for i, ap in enumerate(self.ap_evaluator_q):
                ap.evaluate(
                    metadata["thing_classes"],
                    filedir,
                    load_collect_pred,
                    logger=self.logger,
                    to_file=True,
                    info_name=f"ap_info_{i}",
                    score_name=f"ap_scores_{i}",
                    table_name=f"ap_table_{i}",
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


class CQRConformalDataCollector(DataCollector):
    """
    Subclass of DataCollector for the CQRConformal risk control procedure.
    The stored information is customized to each conformal procedure, e.g.,
    on storing of the conformal scores.
    """
    def __init__(
        self,
        quantiles: list,
        nr_class: int,
        nr_img: int,
        dict_fields: list = [],
        logger=None,
        label_set_generator=None,
    ):
        if not dict_fields:
            dict_fields = _default_dict_fields.copy()
            self.coord_fields = ["x0", "y0", "x1", "y1"]
            # Quantile predictions
            self.quantiles = quantiles
            self.quant_fields = [
                f"pred_q{int(q*100)}_{c}"
                for (q, c) in list(product(self.quantiles, self.coord_fields))
            ]
            # Conformal scores
            self.score_fields = [
                "quant_res_x0",
                "quant_res_y0",
                "quant_res_x1",
                "quant_res_y1",
            ]
            self.nr_scores = len(self.score_fields)
            dict_fields += self.quant_fields
            dict_fields += self.score_fields
        super().__init__(nr_class, nr_img, dict_fields, logger, label_set_generator)

    def __call__(
        self,
        gt_box: Boxes,
        gt_class: torch.Tensor,
        pred_box: Boxes,
        pred_box_q: List[Boxes],
        q_idx: List,
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

            # Add quantile predictions
            for i, f in enumerate(self.quant_fields):
                self.ist_list[c][f] += pred_box_q[i // 4][idx].tensor[:, i % 4].tolist()

            # Add conformal scores
            # Note: q_idx identifies the lower and upper quant preds used for conformal scores
            self.ist_list[c]["quant_res_x0"] += quant_res(
                gt_box[idx].tensor[:, 0],
                pred_box_q[q_idx[0]][idx].tensor[:, 0],
                pred_box_q[q_idx[1]][idx].tensor[:, 0],
            ).tolist()
            self.ist_list[c]["quant_res_y0"] += quant_res(
                gt_box[idx].tensor[:, 1],
                pred_box_q[q_idx[0]][idx].tensor[:, 1],
                pred_box_q[q_idx[1]][idx].tensor[:, 1],
            ).tolist()
            self.ist_list[c]["quant_res_x1"] += quant_res(
                gt_box[idx].tensor[:, 2],
                pred_box_q[q_idx[0]][idx].tensor[:, 2],
                pred_box_q[q_idx[1]][idx].tensor[:, 2],
            ).tolist()
            self.ist_list[c]["quant_res_y1"] += quant_res(
                gt_box[idx].tensor[:, 3],
                pred_box_q[q_idx[0]][idx].tensor[:, 3],
                pred_box_q[q_idx[1]][idx].tensor[:, 3],
            ).tolist()

        if verbose:
            print(f"Added all instances for image {img_id}.")
