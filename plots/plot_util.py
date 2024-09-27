"""
This script contains functions to plot images with ground truth, predictions,
and prediction intervals. The functions are designed to work with Detectron2 
structures and our conformal approaches and are used to visualize results in plots.ipynb.
- d2_plot_pi: plot image with ground truth, predictions, and prediction intervals
- d2_plot_gt: plot image with ground truth annotations
- d2_plot_pred: plot image with predicted annotations

NOTE on channel ordering:
COCO images use cv2 and come in BGR format, but Visualizer uses matplotlib and
thus works with RGB values, so using im[:, :, ::-1] switches the B and R
channel orders (and back). This is important for consistent colouring.

NOTE on cv2_imshow:
This function is available in the Google Colab package and is used to display images
in the notebook via script calls. You can experiment with using cv2.imshow instead.
"""

import torch
import cv2
import matplotlib.colors as mcolors
from google.colab.patches import cv2_imshow

from detectron2.structures import Boxes, Instances
from plots.visualizer import Visualizer, VisualizerFill
from calibration import pred_intervals


def d2_plot_pi(
    risk_control: str,
    image: dict,
    gt_box: Boxes,
    pred: Instances,
    quant: torch.Tensor,
    channel_order: str,
    draw_labels: list = [],
    colors: list = ["red", "green", "palegreen"],
    alpha: list = [0.8, 0.4, 0.4],
    lw: float = None,
    notebook: bool = True,
    to_file: bool = True,
    filename: str = "",
    label_gt: list = None,
    label_set: list = None,
):
    """
    Plot image with ground truth, predictions, and prediction intervals.
    
    Args:
        risk_control (str): risk control procedure, one of ["std_conf", "ens_conf", "cqr_conf"]
        image (dict): image dictionary from Detectron2 dataset
        gt_box (Boxes): ground truth bounding boxes
        pred (Instances): predicted bounding boxes
        quant (torch.Tensor): quantile values for prediction intervals
        channel_order (str): channel order of the image, one of ["BGR", "RGB"]
        draw_labels (list[str]): labels to draw, a subset of ["label_gt", "label_set", "scores"]
        colors (list[matplotlib.colors]): a list of colors, where colors correspond to [ground truth, PI bounds, PI fill]
        alpha (list[float]): a list of alpha values, where alpha values correspond to [ground truth, PI bounds, PI fill]
        lw: linewidth for the drawn bounding boxes
        notebook (bool): display image in notebook
        to_file (bool): save image to file
        filename (str): filename to save image to
        label_gt (list): ground truth labels
        label_set (list): predicted labels
    
    Returns:
        Displays or saves image with ground truth, predictions, and prediction intervals.
    """    

    img_h, img_w = image["height"], image["width"]
    img_path = image["file_name"]
    img = cv2.imread(img_path)
    print(f"Displaying image '{img_path}'.")

    colors = [mcolors.to_rgb(c) for c in colors]

    if channel_order == "BGR":
        img = img[:, :, ::-1]
    elif channel_order == "RGB":
        colors = [c[::-1] for c in colors]

    viz = VisualizerFill(img, scale=1)

    if risk_control == "std_conf":
        boxes, box_ids, box_colors = _get_boxes_std(
            gt_box, pred, quant, img_h, img_w, colors[:2]
        )
    elif risk_control == "ens_conf":
        boxes, box_ids, box_colors = _get_boxes_ens(
            gt_box, pred, quant, img_h, img_w, colors[:2]
        )
    elif risk_control == "cqr_conf":
        boxes, box_ids, box_colors = _get_boxes_cqr(
            gt_box, pred, quant, img_h, img_w, colors[:2]
        )
    else:
        raise ValueError("Invalid risk control procedure.")

    labels = _create_text_labels(label_gt, label_set, pred.scores, draw_labels)
    if labels is not None:  # make it the same length as nr_boxes
        labels = labels + [x for item in labels for x in (item, item)]

    out = viz.overlay_instances(
        boxes=boxes,
        box_ids=box_ids,
        labels=labels,
        assigned_colors=box_colors,
        fill_color=colors[2],
        alpha=alpha,
        lw=lw,
    )
    out_img = out.get_image()

    if channel_order == "BGR":
        out_img = out_img[:, :, ::-1]

    if notebook:
        cv2_imshow(out_img)
    if to_file:
        cv2.imwrite(filename, out_img)
        print(f"Written image to {filename}.")


def _get_boxes_std(gt_box, pred, quant, img_h, img_w, colors):
    # Prepare collection
    nr_gt, nr_pred = len(gt_box), len(pred.pred_boxes)
    nr_boxes = nr_gt + nr_pred * 2
    boxes = torch.empty(size=(nr_boxes, 4), dtype=torch.float32)
    box_colors = []
    box_ids = []

    # add gt boxes
    boxes[:nr_gt] = gt_box.tensor
    box_colors += [colors[0]] * nr_gt
    box_ids += [0] * nr_gt

    # add pred and PI boxes
    for i in range(nr_pred):
        j = nr_gt + i * 2

        predb = pred.pred_boxes.tensor[i]
        q = quant[i]

        pi = pred_intervals.fixed_pi(predb, q)
        # lower PI bound: x0+q, y0+q, x1-q, y1-q
        lower = torch.tensor([pi[0, 1], pi[1, 1], pi[2, 0], pi[3, 0]])
        # upper PI bound: x0-q, y0-q, x1+q, y1+q
        upper = torch.tensor([pi[0, 0], pi[1, 0], pi[2, 1], pi[3, 1]])
        # clip to img size and store
        boxes[j], boxes[j + 1] = _d2_limit_pi_bounds(predb, lower, upper, img_h, img_w)

        box_ids += [i + 1, i + 1]
    box_colors += [colors[1], colors[1]] * nr_pred

    return boxes, box_ids, box_colors


def _get_boxes_ens(gt_box, pred, quant, img_h, img_w, colors):
    # Prepare collection
    nr_gt, nr_pred = len(gt_box), len(pred.pred_boxes)
    nr_boxes = nr_gt + nr_pred * 2
    boxes = torch.empty(size=(nr_boxes, 4), dtype=torch.float32)
    box_colors = []
    box_ids = []

    # add gt boxes
    boxes[:nr_gt] = gt_box.tensor
    box_colors += [colors[0]] * nr_gt
    box_ids += [0] * nr_gt

    # add pred and PI boxes
    for i in range(nr_pred):
        j = nr_gt + i * 2

        predb = pred.pred_boxes.tensor[i]
        unc = pred.unc[i]
        q = quant[i]

        pi = pred_intervals.norm_pi(predb, unc, q)
        # lower PI bound: x0+q*unc, y0+q*unc, x1-q*unc, y1-q*unc
        lower = torch.tensor([pi[0, 1], pi[1, 1], pi[2, 0], pi[3, 0]])
        # upper PI bound: x0-q*unc, y0-q*unc, x1+q*unc, y1+q*unc
        upper = torch.tensor([pi[0, 0], pi[1, 0], pi[2, 1], pi[3, 1]])
        # clip to img size and store
        boxes[j], boxes[j + 1] = _d2_limit_pi_bounds(predb, lower, upper, img_h, img_w)

        box_ids += [i + 1, i + 1]
    box_colors += [colors[1], colors[1]] * nr_pred

    return boxes, box_ids, box_colors


def _get_boxes_cqr(gt_box, pred, quant, img_h, img_w, colors):
    # Prepare collection
    nr_gt, nr_pred = len(gt_box), len(pred.pred_boxes)
    nr_boxes = nr_gt + nr_pred * 2
    boxes = torch.empty(size=(nr_boxes, 4), dtype=torch.float32)
    box_colors = []
    box_ids = []

    # add gt boxes
    boxes[:nr_gt] = gt_box.tensor
    box_colors += [colors[0]] * nr_gt
    box_ids += [0] * nr_gt

    # add pred and PI boxes
    for i in range(nr_pred):
        j = nr_gt + i * 2

        predb = pred.pred_boxes.tensor[i]
        pred_l = pred.pred_lower.tensor[i]
        pred_u = pred.pred_upper.tensor[i]
        q = quant[i]

        pi = pred_intervals.quant_pi(pred_l, pred_u, q)
        # lower PI bound: x0_l+q, y0_l+q, x1_l-q, y1_l-q
        lower = torch.tensor([pi[0, 1], pi[1, 1], pi[2, 0], pi[3, 0]])
        # upper PI bound: x0_h-q, y0_h-q, x1_h+q, y1_h+q
        upper = torch.tensor([pi[0, 0], pi[1, 0], pi[2, 1], pi[3, 1]])
        # clip to img size and store
        boxes[j], boxes[j + 1] = _d2_limit_pi_bounds(predb, lower, upper, img_h, img_w)

        box_ids += [i + 1, i + 1]
    box_colors += [colors[1], colors[1]] * nr_pred

    return boxes, box_ids, box_colors


def _d2_limit_pi_bounds(pred, lower, upper, img_h, img_w):
    # limit lower PI to mid of pred box, upper PI to image size for visualization
    pred_mid_w = pred[0] + 0.5 * (pred[2] - pred[0])  # x0 + 1/2(x1-x0)
    pred_mid_h = pred[1] + 0.5 * (pred[3] - pred[1])  # y0 + 1/2(y1-y0)

    lower = torch.tensor(
        [
            lower[0].clamp_max(pred_mid_w),
            lower[1].clamp_max(pred_mid_h),
            lower[2].clamp_min(pred_mid_w),
            lower[3].clamp_min(pred_mid_h),
        ]
    )
    upper = torch.tensor(
        [
            upper[0].clamp_min(0.0),
            upper[1].clamp_min(0.0),
            upper[2].clamp_max(img_w),
            upper[3].clamp_max(img_h),
        ]
    )
    return lower, upper


def _create_text_labels(label_gt, label_set, scores, draw_labels=[]):
    """
    Create text labels for the boxes.
    
    draw_labels is a list of labels that should be drawn and is
    a subset of ["label_gt", "label_set", "scores"]
    """
    labels = None

    if label_gt is not None and "label_gt" in draw_labels:
        labels = [str(i) for i in label_gt]

    if label_set is not None and "label_set" in draw_labels:
        if labels is None:
            labels = ["[" + f'{{{", ".join(item)}}}' + "]" for item in label_set]
        else:
            labels = [
                l + "," + "[" + f'{{{", ".join(i)}}}' + "]"
                for l, i in zip(labels, label_set)
            ]

    if scores is not None and "scores" in draw_labels:
        if labels is None:
            labels = ["{:.0f}".format(s * 100) for s in scores]
        else:
            labels = ["{} ({:.0f})".format(l, s * 100) for l, s in zip(labels, scores)]

    return labels


def d2_plot_gt(
    image: dict,
    meta: dict,
    channel_order: str,
    draw_labels: list,
    colors: list,
    alpha: float = 0.5,
    notebook: bool = True,
    to_file: bool = True,
    filename: str = "",
):
    """
    Plot image with ground truth annotations.
    
    Args:
        image (dict): image dictionary from Detectron2 dataset
        meta (dict): metadata dictionary from Detectron2 dataset
        channel_order (str): channel order of the image, one of ["BGR", "RGB"]
        draw_labels (list[str]): labels to draw, a subset of ["label_gt", "label_set", "scores"]
        colors (list[matplotlib.colors]): a list of colors, where each color
                    corresponds to each mask or box in the image.
        alpha (float): transparency of the annotations
        notebook (bool): display image in notebook
        to_file (bool): save image to file
        filename (str): filename to save image to
    
    Returns:
        Displays or saves image with ground truth annotations.
    """

    img_path = image["file_name"]
    img = cv2.imread(img_path)
    print(f"Displaying GT for image '{img_path}'.")

    if channel_order == "BGR":
        img = img[:, :, ::-1]
    elif channel_order == "RGB":
        colors = [c[::-1] for c in colors]

    viz = Visualizer(img, meta, scale=1)
    out = viz.draw_dataset_dict(
        image, draw_labels=draw_labels, colors=colors, alpha=alpha
    )
    out_img = out.get_image()

    if channel_order == "BGR":
        out_img = out_img[:, :, ::-1]

    if notebook:
        cv2_imshow(out_img)
    if to_file:
        cv2.imwrite(filename, out_img)
        print(f"Written image to {filename}.")


def d2_plot_pred(
    image: dict,
    pred: Instances,
    meta: dict,
    channel_order: str,
    draw_labels: list,
    colors: list,
    alpha: float = 0.5,
    notebook: bool = True,
    to_file: bool = True,
    filename: str = "",
):
    """
    Plot image with predicted annotations.
    
    Args:
        image (dict): image dictionary from Detectron2 dataset
        pred (Instances): predicted bounding boxes
        meta (dict): metadata dictionary from Detectron2 dataset
        channel_order (str): channel order of the image, one of ["BGR", "RGB"]
        draw_labels (list[str]): labels to draw, a subset of ["label_gt", "label_set", "scores"]
        colors (list[matplotlib.colors]): a list of colors, where each color
                    corresponds to each mask or box in the image.
        alpha (float): transparency of the annotations
        notebook (bool): display image in notebook
        to_file (bool): save image to file
        filename (str): filename to save image to
    
    Returns:
        Displays or saves image with predicted annotations.
    """

    img_path = image["file_name"]
    img = cv2.imread(img_path)
    print(f"Displaying preds for image '{img_path}'.")

    if channel_order == "BGR":
        img = img[:, :, ::-1]
    elif channel_order == "RGB":
        colors = [c[::-1] for c in colors]

    viz = Visualizer(img, meta, scale=1)
    out = viz.draw_instance_predictions(
        pred, draw_labels=draw_labels, colors=colors, alpha=alpha
    )
    out_img = out.get_image()

    if channel_order == "BGR":
        out_img = out_img[:, :, ::-1]

    if notebook:
        cv2_imshow(out_img)
    if to_file:
        cv2.imwrite(filename, out_img)
        print(f"Written image to {filename}.")
