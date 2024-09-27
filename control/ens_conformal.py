import os
import torch
from tqdm import tqdm

from detectron2.structures import Boxes, Instances
from detectron2.data.detection_utils import annotations_to_instances

from .abstract_risk_control import RiskControl
from model import matching, model_loader, ensemble_boxes_wbf
from data.data_collector import DataCollector, _default_dict_fields
from calibration import random_split, pred_intervals
from calibration.conformal_scores import norm_res, one_sided_mult_res
from evaluation import metrics, results_table, ap_evaluation
from util.util import set_seed
from control.classifier_sets import get_label_set_generator


class EnsConformal(RiskControl):
    """
    This class is used to run the conformal bounding box procedure with
    normalized residual nonconformity scores leveraging an object detector
    ensemble (Box-Ens).
    
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
        self.calib_one_sided = cfg.CALIBRATION.ONE_SIDED_PI
        self.calib_alpha = args.alpha

        self.cfg = cfg
        self.ens_size = cfg.MODEL.ENSEMBLE.SIZE
        self.ens_weights = cfg.MODEL.ENSEMBLE.WEIGHTS
        self.min_detects = cfg.MODEL.ENSEMBLE.MIN_DETECTS
        if cfg.MODEL.ENSEMBLE.PARAMS:
            self.ensemble = self._load_param_ensemble(eval=True)
        else:
            self.ensemble = self._load_ensemble(eval=True)

        self.ap_eval = cfg.MODEL.AP_EVAL
        if self.ap_eval:
            self.ap_evaluator = [
                ap_evaluation.APEvaluator(nr_class=nr_class)
                for i in range(self.ens_size)
            ]

        self.label_alpha = args.label_alpha
        self.label_set_generator = get_label_set_generator(cfg, args, logger)

    def set_collector(self, nr_class: int, nr_img: int, dict_fields: list = []):
        """
        Initializes the data collector for the risk control procedure.
        """
        self.collector = EnsConformalDataCollector(
            nr_class,
            nr_img,
            dict_fields,
            self.calib_one_sided,
            self.logger,
            self.label_set_generator,
        )

    def raw_prediction(self, model, img):
        """
        Runs the model on a single image and returns the raw predictions 
        after the ensemble's weighted box fusion.
        """
        w, h = img["width"], img["height"]

        with torch.no_grad():
            pred_boxes, pred_classes, pred_scores, pred_score_all = [], [], [], []

            for m, ens_model in enumerate(self.ensemble):
                pred = ens_model([img])
                ist = pred[0]["instances"]
                # normalize boxes for weighted box fusion (wbf)
                box_norm = torch.div(ist.pred_boxes.tensor, torch.tensor([w, h, w, h]))
                pred_boxes.append(box_norm.tolist())
                pred_classes.append(ist.pred_classes.tolist())
                pred_scores.append(ist.scores.tolist())
                pred_score_all.append(ist.scores_all.tolist())

            # wbf, modified to also return ensemble uncertainty
            boxes, scores, score_all, classes, unc = ensemble_boxes_wbf.weighted_boxes_fusion(
                pred_boxes, pred_scores, pred_score_all, pred_classes
            )

            box_unnorm = torch.tensor(boxes) * torch.tensor([w, h, w, h])
            unc_unnorm = torch.tensor(unc) * torch.tensor([w, h, w, h])
            # replace zero values with one, i.e., recover absolute residuals with std_dev = 1
            unc_unnorm = torch.where(unc_unnorm == 0, torch.tensor(1.0), unc_unnorm)

            ens_ist = Instances((h, w))
            ens_ist.set("pred_boxes", Boxes(box_unnorm))
            ens_ist.set("pred_classes", torch.tensor(classes).to(torch.int))
            ens_ist.set("scores", torch.tensor(scores))
            ens_ist.set("scores_all", torch.tensor(score_all))
            ens_ist.set("unc", unc_unnorm)
        return ens_ist

    def _load_param_ensemble(self, eval: bool = True):
        """
        Load an ensemble of the same model architecture with different
        hyperparameter settings on NMS_THRESH_TEST and SCORE_THRESH_TEST
        for inference. We experimented with this 'hyperparameter ensemble'
        but did not find any significant results.
        """
        seed = self.seed
        ensemble = []

        for i in range(self.ens_size):
            seed += 10000
            set_seed(seed, self.logger)
            self.cfg.MODEL.CONFIG.SEED = seed
            self.cfg.MODEL.CONFIG.MODEL.DEVICE = self.device
            self.cfg.MODEL.DEVICE = self.device

            # set correct parameter pointers
            params = self.cfg.MODEL.ENSEMBLE.PARAMS_VALUES[i]
            self.cfg.MODEL.CONFIG.MODEL.ROI_HEADS.NMS_THRESH_TEST = params[0]
            self.cfg.MODEL.CONFIG.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params[1]
            self.logger.info(f"Params | NMS: {params[0]}, SCORE: {params[1]}")

            cfg_model, model = model_loader.d2_build_model(self.cfg, self.logger)
            model_loader.d2_load_model(cfg_model, model, self.logger)
            if eval:
                model.eval()
            ensemble.append(model)
        return ensemble

    def _load_ensemble(self, eval: bool = True):
        """
        Load an ensemble of models with different architectures but the same
        inference parameter settings, i.e., NMS_THRESH_TEST and SCORE_THRESH_TEST fixed.
        This is the classical ensemble approach also used in our experiments.
        """
        seed = self.seed
        ensemble = []

        for i in range(self.ens_size):
            seed += 10000
            set_seed(seed, self.logger)
            self.cfg.MODEL.CONFIG.SEED = seed
            self.cfg.MODEL.CONFIG.MODEL.DEVICE = self.device
            self.cfg.MODEL.DEVICE = self.device

            # set correct model pointers
            self.cfg.MODEL.NAME = self.cfg.MODEL.ENSEMBLE.NAME[i]
            self.cfg.MODEL.ID = self.cfg.MODEL.ENSEMBLE.ID[i]
            self.cfg.MODEL.FILE = self.cfg.MODEL.ENSEMBLE.FILE[i]

            cfg_model, model = model_loader.d2_build_model(self.cfg, self.logger)
            model_loader.d2_load_model(cfg_model, model, self.logger)
            if eval:
                model.eval()
            ensemble.append(model)
        return ensemble

    def _fuse_ensemble(
        self, gt_nr, pred_boxes, pred_classes, pred_scores, pred_scores_all, weights
    ):
        """
        This method follows the 'weighted box fusion' approach from
        Solovyev et al. (2021). We fuse the ensemble predictions for each ground truth
        instance by weighted mean for box coordinates and mean for class
        and score predictions. The uncertainty is calculated as the standard
        deviation of the ensemble predictions. The method returns the fused
        predictions and the ground truths that have received enough ensemble
        predictions.
        """
        pred_b, pred_u, pred_c, pred_s, pred_s_all = [], [], [], [], []
        gt_mask = torch.zeros(gt_nr, dtype=torch.bool)

        for n in range(gt_nr):
            # get non-negative mask, i.e. all ensemble preds
            mask = pred_scores[n] >= 0
            ens_nr = mask.sum()

            b = pred_boxes[n, :, mask]
            c = pred_classes[n, mask]
            s = pred_scores[n, mask]
            s_all = pred_scores_all[n, :, mask]

            if ens_nr <= self.min_detects:
                # this gt has not received enough ensemble preds
                continue
            else:
                # calculate weights
                if weights == "score":
                    w = s / s.sum()
                else:  # equal weights
                    w = torch.ones((ens_nr,)) / ens_nr

                # fuse box coordinates by weighted mean
                assert b.shape[-1] == w.shape.numel()
                b_fuse = (b * w).sum(dim=-1)
                pred_b.append(b_fuse.tolist())

                # ensemble uncertainty
                unc = b.std(dim=-1)
                # replace zero values with one, i.e. recover abs res (no uncertainty)
                unc = torch.where(unc == 0, torch.tensor(1.0), unc)
                pred_u.append(unc.tolist())

                # fuse pred classes by mode
                pred_c.append(c.mode().values.item())
                # fuse pred scores by mean score
                pred_s.append(s.mean().item())
                pred_s_all.append(s_all.mean(dim=-1).tolist())
                # mark gt as visited
                gt_mask[n] = True

        return (
            Boxes(torch.tensor(pred_b)),
            torch.tensor(pred_u),
            torch.tensor(pred_c),
            torch.tensor(pred_s),
            torch.tensor(pred_s_all),
            gt_mask,
        )

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

        del model

        with torch.no_grad(), tqdm(dataloader, desc="Images") as loader:
            for i, img in enumerate(loader):
                # BoxMode.XYWH to BoxMode.XYXY and correct formatting
                gt = annotations_to_instances(
                    img[0]["annotations"], (img[0]["height"], img[0]["width"])
                )
                gt_box_tr, gt_class_tr = gt.gt_boxes, gt.gt_classes
                gt_nr = len(gt_box_tr)

                # storage of ensemble preds for all gt
                pred_boxes = torch.full((gt_nr, 4, self.ens_size), -1.0)
                pred_scores = torch.full((gt_nr, self.ens_size), -1.0)
                pred_scores_all = torch.full(
                    (gt_nr, self.nr_class, self.ens_size), -1.0
                )
                pred_classes = torch.full((gt_nr, self.ens_size), -1.0)

                for m, ens_model in enumerate(self.ensemble):
                    pred = ens_model(img)
                    pred_ist = pred[0]["instances"].to("cpu")
                    pred_box = pred_ist.pred_boxes
                    pred_class = pred_ist.pred_classes
                    pred_score = pred_ist.scores
                    pred_score_all = pred_ist.scores_all

                    # Collect and store AP eval info
                    if self.ap_eval:
                        self.ap_evaluator[m].collect(
                            gt_box_tr, pred_box, gt_class_tr, pred_class, pred_score
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
                        _,  # matches
                    ) = matching.matching(
                        gt_box_tr,
                        pred_box,
                        gt_class_tr,
                        pred_class,
                        pred_score,
                        pred_score_all,
                        box_matching=self.box_matching,
                        class_match=self.class_matching,
                        thresh=self.iou_thresh,
                    )

                    # collecting ensemble preds
                    mask = (
                        (gt_box_tr.tensor.unsqueeze(1) == gt_box.tensor)
                        .all(dim=-1)
                        .sum(dim=-1)
                        .to(torch.bool)
                    )
                    pred_boxes[mask, :, m] = pred_box.tensor
                    pred_scores[mask, m] = pred_score
                    pred_scores_all[mask, :, m] = pred_score_all
                    pred_classes[mask, m] = pred_class.to(torch.float)

                    del pred, pred_ist

                # fusing ensemble preds
                (
                    pred_box,
                    pred_unc,
                    pred_class,
                    pred_score,
                    pred_score_all,
                    gt_mask,
                ) = self._fuse_ensemble(
                    gt_nr,
                    pred_boxes,
                    pred_classes,
                    pred_scores,
                    pred_scores_all,
                    self.ens_weights,
                )
                gt_box = gt_box_tr[gt_mask]
                gt_class = gt_class_tr[gt_mask]

                # Collect and store risk control info
                if not (gt_mask.sum() == 0):
                    self.collector(
                        gt_box,
                        gt_class,
                        pred_box,
                        pred_unc,
                        pred_score,
                        pred_score_all,
                        img_id=i,
                    )

                if self.log is not None:
                    self.log.define_metric("nr_matches", summary="mean")
                    self.log.log({"nr_matches": len(gt_class)})

                if verbose:
                    self.logger.info(f"\n{gt_class=}\n{pred_class=},\n{pred_score=}")

                del gt, pred_boxes, pred_scores, pred_scores_all, pred_classes

        if self.ap_eval:
            for i in range(self.ens_size):
                self.ap_evaluator[i].to_file(
                    self.ap_evaluator[i].ap_info, f"ap_info_{i}", self.filedir
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
            for i in range(self.ens_size):
                self.ap_evaluator[i].evaluate(
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


class EnsConformalDataCollector(DataCollector):
    """
    Subclass of DataCollector for the EnsConformal risk control procedure.
    The stored information is customized to each conformal procedure, e.g.,
    on storing of the conformal scores.
    """
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

    def __call__(
        self,
        gt_box: Boxes,
        gt_class: torch.Tensor,
        pred_box: Boxes,
        pred_unc: torch.Tensor,
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

            # Add uncertainty scores
            self.ist_list[c]["unc_x0"] += pred_unc[idx, 0].tolist()
            self.ist_list[c]["unc_y0"] += pred_unc[idx, 1].tolist()
            self.ist_list[c]["unc_x1"] += pred_unc[idx, 2].tolist()
            self.ist_list[c]["unc_y1"] += pred_unc[idx, 3].tolist()

            # Add conformal scores
            if self.one_sided:  # one-sided (upper) PI only
                self.ist_list[c]["norm_res_x0"] += one_sided_mult_res(
                    gt_box[idx].tensor[:, 0],
                    pred_box[idx].tensor[:, 0],
                    mult=pred_unc[idx, 0],
                    min=True,
                ).tolist()
                self.ist_list[c]["norm_res_y0"] += one_sided_mult_res(
                    gt_box[idx].tensor[:, 1],
                    pred_box[idx].tensor[:, 1],
                    mult=pred_unc[idx, 1],
                    min=True,
                ).tolist()
                self.ist_list[c]["norm_res_x1"] += one_sided_mult_res(
                    gt_box[idx].tensor[:, 2],
                    pred_box[idx].tensor[:, 2],
                    mult=pred_unc[idx, 2],
                    min=False,
                ).tolist()
                self.ist_list[c]["norm_res_y1"] += one_sided_mult_res(
                    gt_box[idx].tensor[:, 3],
                    pred_box[idx].tensor[:, 3],
                    mult=pred_unc[idx, 3],
                    min=False,
                ).tolist()
            else:
                self.ist_list[c]["norm_res_x0"] += norm_res(
                    gt_box[idx].tensor[:, 0],
                    pred_box[idx].tensor[:, 0],
                    pred_unc[idx, 0],
                ).tolist()
                self.ist_list[c]["norm_res_y0"] += norm_res(
                    gt_box[idx].tensor[:, 1],
                    pred_box[idx].tensor[:, 1],
                    pred_unc[idx, 1],
                ).tolist()
                self.ist_list[c]["norm_res_x1"] += norm_res(
                    gt_box[idx].tensor[:, 2],
                    pred_box[idx].tensor[:, 2],
                    pred_unc[idx, 2],
                ).tolist()
                self.ist_list[c]["norm_res_y1"] += norm_res(
                    gt_box[idx].tensor[:, 3],
                    pred_box[idx].tensor[:, 3],
                    pred_unc[idx, 3],
                ).tolist()

        if verbose:
            print(f"Added all instances for image {img_id}.")
