"""
This script computes different AP metrics for object detection evaluation. 
We follow the COCO evaluation metrics, but reimplement them from scratch to
align with our data formats. The code is fairly vebose and could benefit from
some refactoring or simplification.
"""

import os

import torch
import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from detectron2.structures.boxes import Boxes, pairwise_iou

from util.io_file import save_json, load_json
from util import util


_ap_info_dict_fields = [
    "pred_area",
    "pred_score",
    "tp", "fp",
    "nr_gt",
    "gt_area"
]

_ap_score_dict_fields = [
    "AP",
    "prec",
    "inter_prec",
    "rec",
    "inter_rec",
    "mean_pred_area",
    "mean_gt_area",
    "AP_small",
    "prec_small",
    "rec_small",
    "AP_medium",
    "prec_medium",
    "rec_medium",
    "AP_large",
    "prec_large",
    "rec_large",
    "nr_gt",
]

_ap_score_sizes = {
    "small": [0**2, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e5**2],
}

_ap_metrics = [
    "AP@IOU=.50:.05:.95",
    "AP@IOU=.75",
    "AP@IOU=.50",
    "AR@IOU=.50",
    "Mean pred area",
    "Mean gt area",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_small@IOU=.50",
    "AR_medium@IOU=.50",
    "AR_large@IOU=.50",
    "nr gt",
]


class APEvaluator:
    def __init__(
        self,
        nr_class,
        ap_info_dict_fields: list = _ap_info_dict_fields,
        ap_score_dict_fields: list = _ap_score_dict_fields,
        ap_score_sizes: dict = _ap_score_sizes,
        ap_metrics: list = _ap_metrics,
    ):
        self.nr_class = nr_class
        self.ap_score_dict_fields = ap_score_dict_fields
        self.ap_score_sizes = ap_score_sizes
        self.ap_metrics = ap_metrics

        # AP at IoU=.50:.05:.95
        self.iou_thresholds = torch.arange(0.5, 0.95 + 1e-5, step=0.05)
        self.str_thresholds = [f"AP{int(i*100)}" for i in self.iou_thresholds]

        # {AP score: [info_dict_fields for each class]}
        self.ap_info = {
            s: [{k: [] for k in ap_info_dict_fields} for _ in range(nr_class)]
            for s in self.str_thresholds
        }
        # {AP score: [ap_score_dict_fields for each class]}
        self.ap_scores = {
            s: [{k: [] for k in ap_score_dict_fields} for _ in range(nr_class)]
            for s in self.str_thresholds
        }

    def collect(
        self,
        gt_box: Boxes,
        pred_box: Boxes,
        gt_class: torch.Tensor,
        pred_class: torch.Tensor,
        pred_score: torch.Tensor,
    ):
        for i, iou_thresh in enumerate(self.iou_thresholds):
            # degenerate case: no gt or pred boxes
            if (len(gt_box) == 0) or (len(pred_box) == 0):
                continue

            # init empty stores
            s = self.str_thresholds[i]
            gt_match_idx = torch.full((len(gt_box),), fill_value=-1)
            tp, fp = torch.zeros(len(pred_box)), torch.zeros(len(pred_box))

            # assign pred to gt by max iou matching
            pair_iou = pairwise_iou(gt_box, pred_box)

            iou_val, gt_idx = pair_iou.max(dim=0)

            for j in torch.unique(gt_idx, sorted=True):
                # get iou_max for all pred assigned to that gt
                iou_max = torch.max(iou_val[gt_idx == j])
                # get idx of iou_max in iou_val
                # what if two iou scores have identical values?
                # it doesn't matter, since we care about counts
                iou_max_idx = torch.where(iou_max == iou_val)[0]

                if iou_max_idx.numel() == 0:  # degenerate case: empty idx
                    continue

                if iou_max_idx.size() == torch.Size([]):  # not correct sizing
                    iou_max_idx = iou_max_idx.unsqueeze(0)

                # check if iou_max above threshold
                # if yes then mark gt as TP matched with iou_max_idx
                if iou_max >= iou_thresh:
                    gt_match_idx[j] = iou_max_idx

            # Mark TP and FP
            tp[gt_match_idx[gt_match_idx != -1]] = 1
            fp[tp != 1] = 1

            # Store AP-relevant info for all pred and gt boxes
            for c in torch.unique(pred_class).numpy():
                idx = torch.nonzero(pred_class == c, as_tuple=True)[0]
                self.ap_info[s][c]["pred_area"] += pred_box[idx].area().tolist()
                self.ap_info[s][c]["pred_score"] += pred_score[idx].tolist()
                self.ap_info[s][c]["tp"] += tp[idx].tolist()
                self.ap_info[s][c]["fp"] += fp[idx].tolist()

            for cc in torch.unique(gt_class).numpy():
                self.ap_info[s][cc]["nr_gt"] += [(gt_class == cc).sum().item()]
                self.ap_info[s][cc]["gt_area"] += gt_box[gt_class == cc].area().tolist()

    def _compute(self):
        for i in tqdm(range(len(self.iou_thresholds)), desc="IoU thresh"):
            s = self.str_thresholds[i]

            for c in range(self.nr_class):
                # get correct ap_info dict values as tensor
                info = {k: torch.tensor(v) for k, v in self.ap_info[s][c].items()}

                # sort by descending pred_score values
                sort_idx = torch.argsort(info["pred_score"], descending=True)
                nr_gt = info["nr_gt"].sum()
                # get cumulative TP and FP counts
                tp_acc = torch.cumsum(info["tp"][sort_idx], dim=0)
                fp_acc = torch.cumsum(info["fp"][sort_idx], dim=0)

                # Calculate recall (rec) and precision (prec)
                rec = (tp_acc / nr_gt).numpy()  # rec = TP/(TP+FN)
                prec = torch.div(tp_acc, (tp_acc + fp_acc)).numpy()  # prec = TP/(TP+FP)

                # Degenerate case: empty info
                rec = np.append(rec, 0.0) if 0 in rec.shape else rec
                prec = np.append(prec, 0.0) if 0 in prec.shape else prec

                # calculate AP using every-point interpolation of PR curve
                ap, mprec, mrec, _ = self._average_precision(rec, prec)

                # Add scores
                self.ap_scores[s][c]["AP"] += [ap]
                self.ap_scores[s][c]["prec"] += prec.tolist()
                self.ap_scores[s][c]["inter_prec"] += np.array(
                    mprec
                ).tolist()  # interpolated precision
                self.ap_scores[s][c]["rec"] += rec.tolist()
                self.ap_scores[s][c]["inter_rec"] += np.array(
                    mrec
                ).tolist()  # interpolated recall

                # Add mean area sizes (sqrt to express as area**2 patches)
                self.ap_scores[s][c]["mean_pred_area"] += [
                    info["pred_area"].mean().sqrt().item()
                ]
                self.ap_scores[s][c]["mean_gt_area"] += [
                    info["gt_area"].mean().sqrt().item()
                ]
                self.ap_scores[s][c]["nr_gt"] += [nr_gt.float().item()]

                # calculating AP for different sized objects
                for k, v in self.ap_score_sizes.items():
                    ap_s, prec_s, rec_s = self._average_precision_by_size(info, size=v)
                    self.ap_scores[s][c][f"AP_{k}"] += [ap_s]
                    self.ap_scores[s][c][f"prec_{k}"] += prec_s.tolist()
                    self.ap_scores[s][c][f"rec_{k}"] += rec_s.tolist()

    def _average_precision(self, rec, prec):
        """
        This implementation follows
        https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
        """
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)

        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1 + i] != mrec[i]:
                ii.append(i + 1)

        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    def _average_precision_by_size(self, info: dict, size: list):
        # filter instances by desired size
        area_mask = (info["pred_area"] > size[0]) * (info["pred_area"] <= size[1])
        nr_gt = ((info["gt_area"] > size[0]) * (info["gt_area"] <= size[1])).sum()

        # AP calculation using only those filtered instances
        sort_idx = torch.argsort(info["pred_score"][area_mask], descending=True)
        tp_acc = torch.cumsum(info["tp"][area_mask][sort_idx], dim=0)
        fp_acc = torch.cumsum(info["fp"][area_mask][sort_idx], dim=0)
        rec = (tp_acc / nr_gt).numpy()
        prec = torch.div(tp_acc, (tp_acc + fp_acc)).numpy()

        # Degenerate case: empty info
        rec = np.append(rec, 0.0) if 0 in rec.shape else rec
        prec = np.append(prec, 0.0) if 0 in prec.shape else prec

        ap, _, _, _ = self._average_precision(rec, prec)

        return ap, prec, rec

    def to_file(self, file, filename: str, filedir: str, **kwargs):
        fname = os.path.split(filedir)[-1] + f"_{filename}"
        save_json(file, fname, filedir, **kwargs)

    def load_file(self, fname: str, filedir: str, load_collect_pred, logger):
        if load_collect_pred is not None:
            try:
                pred_filedir = os.path.join(
                    os.path.split(filedir)[0], load_collect_pred
                )
                logger.info(f"Trying to load existing {fname} from '{pred_filedir}'.")
                file = load_json(f"{load_collect_pred}_{fname}", pred_filedir)
            except FileNotFoundError:
                logger.info(f"File not found using '{load_collect_pred}'")
                file = None
        else:
            try:
                logger.info(f"Trying to load existing {fname} from '{filedir}'.")
                file = load_json(f"{os.path.split(filedir)[-1]}_{fname}", filedir)
            except FileNotFoundError:
                logger.info(f"File not found using '{filedir}'")
                file = None
        return file

    def _aggregate(self):
        # Tensor with final class-conditional AP metrics
        self.scores = torch.zeros((self.nr_class, len(self.ap_metrics)))

        for c in range(self.nr_class):
            c_scores = torch.zeros(
                (len(self.iou_thresholds), len(self.ap_score_dict_fields))
            )
            # Collect scores for all IoU thresholds for given class
            for i, str_thresh in enumerate(self.str_thresholds):
                for j, str_score in enumerate(self.ap_score_dict_fields):
                    c_scores[i, j] = torch.tensor(
                        self.ap_scores[str_thresh][c][str_score]
                    ).mean()  # Take mean across lists e.g. for prec, rec

            # Fill with final AP metrics for given class
            # NOTE: order of metrics in self.ap_metrics and self.ap_score_dict_fields matters here
            self.scores[c, :] = torch.stack(
                [
                    c_scores[:, 0].mean(),  # AP@IOU=.50:.05:.95
                    c_scores[self.str_thresholds.index("AP75"), 0],  # AP@IOU=.75
                    c_scores[self.str_thresholds.index("AP50"), 0],  # AP@IOU=.50
                    c_scores[self.str_thresholds.index("AP50"), 3].mean(),  # AR@IOU=.50
                    c_scores[:, 5].mean(),  # Mean pred area
                    c_scores[:, 6].mean(),  # Mean gt area
                    c_scores[:, 7].mean(),  # AP@IOU=.50:.05:.95 small
                    c_scores[
                        self.str_thresholds.index("AP50"), 9
                    ].mean(),  # AR@IOU=.50 small
                    c_scores[:, 10].mean(),  # AP@IOU=.50:.05:.95 medium
                    c_scores[
                        self.str_thresholds.index("AP50"), 12
                    ].mean(),  # AR@IOU=.50 medium
                    c_scores[:, 13].mean(),  # AP@IOU=.50:.05:.95 large
                    c_scores[
                        self.str_thresholds.index("AP50"), 15
                    ].mean(),  # AR@IOU=.50 large
                    c_scores[:, 16].mean(),  # nr gt
                ],
                dim=0,
            )

    def evaluate(
        self,
        class_names: list,
        filedir: str,
        load_collect_pred,
        logger,
        to_file: bool = True,
        info_name: str = "ap_info",
        score_name: str = "ap_scores",
        table_name: str = "ap_table",
    ):
        # Get ap_info, otherwise throw error
        ap_info = self.load_file(info_name, filedir, load_collect_pred, logger)
        if ap_info is None:
            raise FileNotFoundError("Require ap_info file for AP eval.")
        else:
            self.ap_info = ap_info

        # Get ap_scores, otherwise compute them
        ap_scores = self.load_file(score_name, filedir, load_collect_pred, logger)
        if ap_scores is None:
            logger.info("Running AP evaluation...")
            self._compute()
            self.to_file(self.ap_scores, score_name, filedir)
        else:
            self.ap_scores = ap_scores

        # Aggregate ap_scores and get into final form
        self._aggregate()

        # Get aggregated scores to results table
        scores_l = self.scores.tolist()
        for i, el in enumerate(scores_l):
            el.insert(0, class_names[i])

        # Add means over class groups
        scores_l.insert(0, ["mean class"] + self.scores.mean(dim=0).tolist())

        samp_bdd = torch.tensor(list(util.get_bdd_as_coco_classes().values()))
        scores_l.insert(
            1, ["mean class (bdd100k)"] + self.scores[samp_bdd].mean(dim=0).tolist()
        )
        scores_l.insert(
            2,
            ["mean class (bdd100k - stop sign)"]
            + self.scores[samp_bdd[:-1]].mean(dim=0).tolist(),
        )

        samp_select = torch.tensor(list(util.get_selected_coco_classes().values()))
        scores_l.insert(
            3, ["mean class (selected)"] + self.scores[samp_select].mean(dim=0).tolist()
        )

        colnames = ["class"] + self.ap_metrics

        filepath = os.path.join(
            filedir, os.path.split(filedir)[-1] + f"_{table_name}.csv"
        )
        if to_file:
            DataFrame(scores_l, columns=colnames).to_csv(
                filepath, index=True, na_rep="NA", float_format="%.4f"
            )
            logger.info(f"Written results to '{filepath}'.")
