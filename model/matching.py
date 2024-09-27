from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor, nonzero
from detectron2.structures.boxes import Boxes, pairwise_iou, matched_pairwise_iou


def d2_box_matching(
    gt_box: Boxes, pred_box: Boxes, thresh: float, verbose: bool = False
):
    """
    Matches ground truth and prediction boxes via Hungarian matching,
    then removes any matches with an IoU below a user-specified
    threshold of confidence. For a single image with potentially
    multiple instances as passed by the dataloader.

    Args:
        gt_box (Boxes): Boxes object with instance-wise ground truth box coordinates.
        pred_box (Boxes): Boxes object with instance-wise predicted box coordinates.
        thresh (float): IoU threshold of confidence to filter for true positives.
        verbose (bool, optional): Print output. Defaults to False.

    Returns:
        (Tensor, Tensor): Indices of filtered GT boxes and respective pred box assignment.
        box_matches (bool): if # of identified matches are non-empty and thus useful.
    """
    assert 0 <= thresh <= 1, f"{thresh=} not in [0,1]"
    pair_iou = pairwise_iou(gt_box, pred_box)  # IoU matrix (gt x pred)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    # Pred in col idx i matches gt in row idx i
    gt_idx, pred_idx = linear_sum_assignment(pair_iou, maximize=True)
    # Sort according to match idx and filter out FP and some FN (gt with no pred)
    gt_matches = gt_box[gt_idx]
    pred_matches = pred_box[pred_idx]
    # Recalculate IoU and filter for TP by considering matches with IoU < thresh as FN
    match_above_thresh = matched_pairwise_iou(gt_matches, pred_matches) >= thresh

    # Only a single instance
    if gt_idx.size == 1 or pred_idx.size == 1:
        match_above_thresh = match_above_thresh.numpy()

    gt_tp_idx = gt_idx[match_above_thresh]
    pred_tp_idx = pred_idx[match_above_thresh]

    # Check if non-empty and thus useful for downstream tasks
    box_matches = False if (gt_idx.size == 0 or pred_idx.size == 0) else True

    if verbose:
        print(f"IOU box matching:\nGT\n{gt_idx}\nPred\n{pred_idx}")
        print(f"\nFiltering for TP with IoU {thresh=}.")
        print(f"Filtered indices:\nGT\n{gt_tp_idx.tolist()}\nPred\n{list(pred_tp_idx)}")

    return gt_tp_idx, pred_tp_idx, box_matches


def d2_max_matching(gt_box: Boxes, pred_box: Boxes, verbose: bool = False):
    """
    Similar to d2_box_matching but uses simple IoU max to
    assign predicted boxes to ground truth boxes.
    Approach maintains all FP boxes.
    """
    pair_iou = pairwise_iou(gt_box, pred_box)  # IoU matrix (gt x pred)
    # assign each pred a gt based on max iou
    if 0 in pair_iou.shape:  # empty in some dim
        gt_idx = torch.tensor([])
    else:
        _, gt_idx = pair_iou.max(dim=0)
    box_matches = False if gt_idx.numpy().size == 0 else True

    if verbose:
        print(
            f"Assigned {pred_box.tensor.shape[0]} pred boxes to "
            f"{gt_box.tensor.shape[0]} gt boxes."
        )

    return gt_idx, box_matches


def class_matching(gt_class: Tensor, pred_class: Tensor, verbose: bool = False):
    """
    Additionally filters out boxes for which predicted and true class label do not match.
    """
    match_idx = nonzero(pred_class == gt_class, as_tuple=True)[0]
    # Check if non-empty and thus useful for downstream tasks
    class_matches = False if len(match_idx) == 0 else True

    if verbose:
        print(f"{len(match_idx)}/{len(gt_class)} box matches are class matches.")
        print(f"Class matching indices: {match_idx}")

    return match_idx, class_matches


def matching(
    gt_box: Boxes,
    pred_box: Boxes,
    gt_class: Tensor,
    pred_class: Tensor,
    pred_score: Tensor,
    pred_score_all=None,
    pred_logits_all=None,
    box_matching: str = "box",
    class_match: bool = False,
    thresh: float = 0.5,
    verbose: bool = False,
    return_idx: bool = False,
):
    """
    Function to match ground truth and predicted boxes and classes.
    
    Args:
        gt_box (Boxes): Ground truth boxes.
        pred_box (Boxes): Predicted boxes.
        gt_class (Tensor): Ground truth classes.
        pred_class (Tensor): Predicted classes.
        pred_score (Tensor): Predicted scores.
        pred_score_all (Tensor, optional): Full predicted score vectors.
        pred_logits_all (Tensor, optional): Full predicted logit vectors.
        box_matching (str, optional): Matching procedure. Defaults to "box".
        class_match (bool, optional): Match classes. Defaults to False.
        thresh (float, optional): IoU threshold. Defaults to 0.5.
        verbose (bool, optional): Print output. Defaults to False.
        return_idx (bool, optional): Return indices. Defaults to False.
    
    Returns:
        (Boxes, Boxes, Tensor, Tensor, Tensor, Tensor, Tensor, bool): 
        Ground truth and predicted boxes, classes, scores, optionally indices of matched boxes.
    """
    matches = False

    if box_matching == "box":  # TP matching
        gt_idx, pred_idx, box_matches = d2_box_matching(
            gt_box, pred_box, thresh, verbose
        )
        gt_box, gt_class = gt_box[gt_idx], gt_class[gt_idx]
        pred_box, pred_class = pred_box[pred_idx], pred_class[pred_idx]
        pred_score = pred_score[pred_idx]
        matches = box_matches

        # if available
        pred_score_all = (
            pred_score_all[pred_idx] if pred_score_all is not None else torch.tensor([])
        )
        pred_logits_all = (
            pred_logits_all[pred_idx]
            if pred_logits_all is not None
            else torch.tensor([])
        )

    elif box_matching == "max":  # TP+FP matching
        gt_idx, box_matches = d2_max_matching(gt_box, pred_box, verbose)
        gt_b = torch.empty(size=(gt_idx.shape[0], 4), dtype=torch.float32)
        gt_c = torch.empty(size=(gt_idx.shape[0],), dtype=torch.int)

        for i, idx in enumerate(gt_idx):
            gt_b[i] = gt_box.tensor[idx]
            gt_c[i] = gt_class[idx]

        gt_box = Boxes(gt_b)
        gt_class = gt_c
        matches = box_matches
        pred_idx = torch.arange(gt_box.tensor.shape[0])

    else:
        raise ValueError("Need to specify valid box matching procedure.")

    if class_match:
        class_idx, class_matches = class_matching(gt_class, pred_class, verbose)
        gt_box, gt_class = gt_box[class_idx], gt_class[class_idx]
        pred_box, pred_class = pred_box[class_idx], pred_class[class_idx]
        pred_score = pred_score[class_idx]
        matches = class_matches

        # if available
        pred_score_all = (
            pred_score_all[class_idx]
            if pred_score_all is not None
            else torch.tensor([])
        )
        pred_logits_all = (
            pred_logits_all[class_idx]
            if pred_logits_all is not None
            else torch.tensor([])
        )
    else:
        class_idx = torch.arange(gt_box.tensor.shape[0])

    return_res = (
        gt_box,
        pred_box,
        gt_class,
        pred_class,
        pred_score,
        pred_score_all,
        pred_logits_all,
        matches,
    )
    if return_idx:
        return_res = (*return_res, gt_idx, pred_idx, class_idx)

    return return_res
