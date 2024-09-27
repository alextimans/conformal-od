"""
This script contains functions to pool the evaluation metrics into a table for easier comparison.
Provided metrics are averaged over the correct indices (e.g., the number of trials), collected in
a pandas DataFrame and saved to a CSV file.

Three functions with similar structure are provided for the three evaluation tasks:
- get_results_table: for bounding box intervals WITHOUT label set construction
- get_box_set_results_table: for bounding box intervals WITH label set construction
- get_label_results_table: for label set construction

These csv files are then used to generate the tables in the paper via results_tables.ipynb.
"""

import os
from pandas import DataFrame
import torch

from util import util


_default_metrics = [
    "nr calib",
    "q x0",
    "q y0",
    "q x1",
    "q y1",
    "mpiw",
    "box stretch",
    "cov x0",
    "cov y0",
    "cov x1",
    "cov y1",
    "cov box",
    "cov box area S",
    "cov box area M",
    "cov box area L",
    "cov box IOU<0.7",
    "cov box IOU[0.7,0.9]",
    "cov box IOU>0.9",
]

# NOTE: indices of metrics in control_data tensor, as fed by the risk control procedure
_idx_metrics = {
    "nr_calib_samp": 0,
    "quant": 1,
    "mpiw": 2,
    "box_stretch": 3,
    "cov_coord": 4,
    "cov_box": 5,
    "cov_area": [6, 9],
    "cov_iou": [9, 12],
}

_default_box_set_metrics = [
    "nr calib",
    "q x0",
    "q y0",
    "q x1",
    "q y1",
    "mpiw",
    "box stretch",
    "cov x0",
    "cov y0",
    "cov x1",
    "cov y1",
    "cov box",
    "cov box area S",
    "cov box area M",
    "cov box area L",
    "cov box IOU<0.7",
    "cov box IOU[0.7,0.9]",
    "cov box IOU>0.9",
    "cov box cl",
    "cov box miscl",
    "mpiw cl",
    "mpiw miscl",
]

# NOTE: indices of metrics in control_data tensor, as fed by the risk control procedure
_idx_box_set_metrics = {
    "nr_calib_samp": 0,
    "quant": 1,
    "mpiw": 2,
    "box_stretch": 3,
    "cov_coord": 4,
    "cov_box": 5,
    "cov_area": [6, 9],
    "cov_iou": [9, 12],
    "cov_box_cl": 12,
    "cov_box_miscl": 13,
    "mpiw_cl": 14,
    "mpiw_miscl": 15,
}

_default_label_metrics = [
    "nr calib",
    "quant",
    "null set frac",
    "mean set size",
    "cov set",
    "cov set area S",
    "cov set area M",
    "cov set area L",
    "cov set IOU<0.7",
    "cov set IOU[0.7,0.9]",
    "cov set IOU>0.9",
    "cov set cl",
    "cov set miscl",
    "mean set size cl",
    "mean set size miscl",
]

# NOTE: indices of metrics in label_data tensor, as fed by the label set procedure
_idx_label_metrics = {
    "nr_calib_samp": 0,
    "quant": 1,
    "null_set_frac": 2,
    "mean_set_size": 3,
    "cov_set": 4,
    "cov_area": [5, 8],
    "cov_iou": [8, 11],
    "cov_cl": [11, 13],
    "mean_set_size_cl": 13,
    "mean_set_size_miscl": 14,
}


def get_results_table(
    data: torch.Tensor,
    class_names: list,
    metrics: list = _default_metrics,
    idx: dict = _idx_metrics,
    to_file: bool = True,
    filename: str = "",
    filedir: str = "",
    logger=None,
):
    # Means over trials
    data = torch.mean(data, dim=0)  # dim (nr_class, 4, nr_metrics)

    # Some metrics also means over coordinates; dim (nr_class)
    mpiw = data[..., idx["mpiw"]].mean(dim=-1, keepdim=True)
    stretch = data[..., idx["box_stretch"]].mean(
        dim=-1, keepdim=True
    )  # mean over 4 identical values
    nr_calib_samp = data[..., idx["nr_calib_samp"]].mean(
        dim=-1, keepdim=True
    )  # mean over 4 identical values
    cov_box = data[..., idx["cov_box"]].mean(
        dim=-1, keepdim=True
    )  # mean over 4 identical values
    cov_area = data[..., idx["cov_area"][0] : idx["cov_area"][1]].mean(
        dim=-2
    )  # mean over 4 identical values
    cov_iou = data[..., idx["cov_iou"][0] : idx["cov_iou"][1]].mean(
        dim=-2
    )  # mean over 4 identical values

    # collect in desired order of metrics
    data = torch.cat(
        [
            nr_calib_samp,
            data[..., idx["quant"]],
            mpiw,
            stretch,
            data[..., idx["cov_coord"]],
            cov_box,
            cov_area,
            cov_iou,
        ],
        dim=1,
    )  # dim (nr_class, len(metrics))

    data_l = data.tolist()
    # Add class names to list
    for i, el in enumerate(data_l):
        el.insert(0, class_names[i])

    # Add means over class groups
    samp_gt0 = torch.where(nr_calib_samp > 0)[0]
    data_l.insert(
        0, ["mean class (nr calib > 0)"] + data[samp_gt0].mean(dim=0).tolist()
    )

    samp_gt100 = torch.where(nr_calib_samp >= 100)[0]
    data_l.insert(
        1, ["mean class (nr calib >= 100)"] + data[samp_gt100].mean(dim=0).tolist()
    )

    samp_gt1000 = torch.where(nr_calib_samp >= 1000)[0]
    data_l.insert(
        2, ["mean class (nr calib >= 1000)"] + data[samp_gt1000].mean(dim=0).tolist()
    )

    samp_bdd = torch.tensor(list(util.get_bdd_as_coco_classes().values()))
    data_l.insert(3, ["mean class (bdd100k)"] + data[samp_bdd].mean(dim=0).tolist())

    samp_select = torch.tensor(list(util.get_selected_coco_classes().values()))
    data_l.insert(4, ["mean class (selected)"] + data[samp_select].mean(dim=0).tolist())

    # Column names
    colnames = ["class"] + metrics

    if to_file:
        filepath = os.path.join(filedir, filename + ".csv")
        DataFrame(data_l, columns=colnames).to_csv(
            filepath, index=True, na_rep="NA", float_format="%.4f"
        )
        logger.info(f"Written results to {filename}.")  # type:ignore


def get_box_set_results_table(
    data: torch.Tensor,
    class_names: list,
    metrics: list = _default_box_set_metrics,
    idx: dict = _idx_box_set_metrics,
    to_file: bool = True,
    filename: str = "",
    filedir: str = "",
    logger=None,
):
    # Means over trials
    data = torch.mean(data, dim=0)  # dim (nr_class, 4, nr_metrics)

    # Some metrics also means over coordinates; dim (nr_class)
    mpiw = data[..., idx["mpiw"]].mean(dim=-1, keepdim=True)
    stretch = data[..., idx["box_stretch"]].mean(
        dim=-1, keepdim=True
    )  # mean over 4 identical values
    nr_calib_samp = data[..., idx["nr_calib_samp"]].mean(
        dim=-1, keepdim=True
    )  # mean over 4 identical values
    cov_box = data[..., idx["cov_box"]].mean(
        dim=-1, keepdim=True
    )  # mean over 4 identical values
    cov_area = data[..., idx["cov_area"][0] : idx["cov_area"][1]].mean(
        dim=-2
    )  # mean over 4 identical values
    cov_iou = data[..., idx["cov_iou"][0] : idx["cov_iou"][1]].mean(
        dim=-2
    )  # mean over 4 identical values

    cov_cl = data[..., idx["cov_box_cl"]].mean(dim=-1, keepdim=True)
    cov_miscl = data[..., idx["cov_box_miscl"]].mean(dim=-1, keepdim=True)
    mpiw_cl = data[..., idx["mpiw_cl"]].mean(dim=-1, keepdim=True)
    mpiw_miscl = data[..., idx["mpiw_miscl"]].mean(dim=-1, keepdim=True)

    # collect in desired order of metrics
    data = torch.cat(
        [
            nr_calib_samp,
            data[..., idx["quant"]],
            mpiw,
            stretch,
            data[..., idx["cov_coord"]],
            cov_box,
            cov_area,
            cov_iou,
            cov_cl,
            cov_miscl,
            mpiw_cl,
            mpiw_miscl,
        ],
        dim=1,
    )  # dim (nr_class, len(metrics))

    data_l = data.tolist()
    # Add class names to list
    for i, el in enumerate(data_l):
        el.insert(0, class_names[i])

    # Add means over class groups
    samp_gt0 = torch.where(nr_calib_samp > 0)[0]
    data_l.insert(
        0, ["mean class (nr calib > 0)"] + data[samp_gt0].mean(dim=0).tolist()
    )

    samp_gt100 = torch.where(nr_calib_samp >= 100)[0]
    data_l.insert(
        1, ["mean class (nr calib >= 100)"] + data[samp_gt100].mean(dim=0).tolist()
    )

    samp_gt1000 = torch.where(nr_calib_samp >= 1000)[0]
    data_l.insert(
        2, ["mean class (nr calib >= 1000)"] + data[samp_gt1000].mean(dim=0).tolist()
    )

    samp_bdd = torch.tensor(list(util.get_bdd_as_coco_classes().values()))
    data_l.insert(3, ["mean class (bdd100k)"] + data[samp_bdd].mean(dim=0).tolist())

    samp_select = torch.tensor(list(util.get_selected_coco_classes().values()))
    data_l.insert(4, ["mean class (selected)"] + data[samp_select].mean(dim=0).tolist())

    # Column names
    colnames = ["class"] + metrics

    if to_file:
        filepath = os.path.join(filedir, filename + ".csv")
        DataFrame(data_l, columns=colnames).to_csv(
            filepath, index=True, na_rep="NA", float_format="%.4f"
        )
        logger.info(f"Written results to {filename}.")  # type:ignore


def get_label_results_table(
    data: torch.Tensor,
    class_names: list,
    metrics: list = _default_label_metrics,
    idx: dict = _idx_label_metrics,
    to_file: bool = True,
    filename: str = "",
    filedir: str = "",
    logger=None,
):
    # Means over trials
    data = torch.mean(data, dim=0)  # dim (nr_class, nr_metrics)
    nr_calib_samp = data[..., idx["nr_calib_samp"]]

    data_l = data.tolist()
    # Add class names to list
    for i, el in enumerate(data_l):
        el.insert(0, class_names[i])

    # Add means over class groups
    samp_gt0 = torch.where(nr_calib_samp > 0)[0]
    data_l.insert(
        0, ["mean class (nr calib > 0)"] + data[samp_gt0].mean(dim=0).tolist()
    )

    samp_gt100 = torch.where(nr_calib_samp >= 100)[0]
    data_l.insert(
        1, ["mean class (nr calib >= 100)"] + data[samp_gt100].mean(dim=0).tolist()
    )

    samp_gt1000 = torch.where(nr_calib_samp >= 1000)[0]
    data_l.insert(
        2, ["mean class (nr calib >= 1000)"] + data[samp_gt1000].mean(dim=0).tolist()
    )

    samp_bdd = torch.tensor(list(util.get_bdd_as_coco_classes().values()))
    data_l.insert(3, ["mean class (bdd100k)"] + data[samp_bdd].mean(dim=0).tolist())

    samp_select = torch.tensor(list(util.get_selected_coco_classes().values()))
    data_l.insert(4, ["mean class (selected)"] + data[samp_select].mean(dim=0).tolist())

    # Column names
    colnames = ["class"] + metrics

    if to_file:
        filepath = os.path.join(filedir, filename + ".csv")
        DataFrame(data_l, columns=colnames).to_csv(
            filepath, index=True, na_rep="NA", float_format="%.4f"
        )
        logger.info(f"Written results to {filename}.")  # type:ignore
