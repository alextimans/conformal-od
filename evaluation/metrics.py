"""
This script contains functions to compute various evaluation metrics for our tasks.
These include:
- Box coverage
- Stratified box coverage by object size and IOU
- Mean Prediction Interval Width (MPIW)
- Box Stretch (used by Andeol et al 2023)
- Label set coverage
- Stratified label set coverage by object size, IOU, and classification
- Mean label set size
- Null set fraction of label sets
- Classifier accuracy
- Classifier's Expected Calibration Error (ECE)
"""

import torch


_size_bins = [[0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
_iou_bins = [[0.0, 0.7], [0.7, 0.9], [0.9, 1.0]]


def coverage(gt: torch.Tensor, pi: torch.Tensor):
    """
    Compute empirical coverage of ground truths in given prediction interval (PI).

    Args:
        gt (Tensor): Ground truth values.
        pi (Tensor): Prediction interval bounds (0: lower, 1: higher).

    Returns:
        Tensor: Individual empirical coverage values on coordinate level
        and on box level per scoring group, outputs for each coordinate-score item.
    """
    covered = (pi[:, 0] <= gt) * (pi[:, 1] >= gt)
    nr_samp = covered.shape[0]
    cov_coord = covered.sum(dim=0) / nr_samp
    # Rearrange tensor for easier box-level compute, e.g. (x, 8) to (x, 2, 4)
    # Assumes that each scoring group of 4 coordinates is grouped together
    covered = covered.view(nr_samp, covered.shape[-1] // 4, 4)
    # Box is covered if every coordinate is covered
    cov_box = (covered.sum(dim=-1) >= 4).sum(dim=0) / nr_samp

    return cov_coord, torch.repeat_interleave(cov_box, 4)  # dims (nr_scores)


def stratified_coverage(
    gt: torch.Tensor,
    pi: torch.Tensor,
    calib_mask: torch.Tensor,
    ist: dict,
    size_bins: list = _size_bins,
    iou_bins: list = _iou_bins,
):
    """
    Compute stratified empirical coverage of ground truths in given prediction interval (PI).
    
    Args:
        gt (Tensor): Ground truth values.
        pi (Tensor): Prediction interval bounds (0: lower, 1: higher).
        calib_mask (Tensor): Calibration mask.
        ist (dict): Dictionary with object instance information.
        size_bins (list): List of area size bins.
        iou_bins (list): List of IOU bins.
        
    Returns:
        Tensor: Empirical coverage values stratified by object area and IOU bins.
    """
    # box cov stratified by GT area bins (same as used in AP eval)
    area = ist["gt_area"]
    cov_area = torch.zeros((gt.shape[-1], len(size_bins)))
    for i, size in enumerate(size_bins):
        mask = (~calib_mask) * (area > size[0]) * (area <= size[1])
        _, cov_area[:, i] = coverage(gt[mask], pi[mask])

    # box cov stratified by IOU gt-pred bins
    ious = ist["iou"]
    cov_iou = torch.zeros((gt.shape[-1], len(iou_bins)))
    for i, iou in enumerate(iou_bins):
        mask = (~calib_mask) * (ious > iou[0]) * (ious <= iou[1])
        _, cov_iou[:, i] = coverage(gt[mask], pi[mask])

    return cov_area, cov_iou


def mean_pi_width(pi: torch.Tensor):
    """
    Returns mean prediction interval width (MPIW), mean over box instances.

    Args:
        pi (Tensor): Prediction interval bounds (0: lower, 1: higher).

    Returns:
        Tensor: Mean prediction interval widths on coordinate level
    """
    return torch.mean(torch.abs(pi[:, 1] - pi[:, 0]), dim=0)


def box_stretch(pi: torch.Tensor, pred_area: torch.Tensor):
    """
    Returns mean box stretch, mean over box instances.
    As used by Andeol et al. (2023), see Eq. 20 in the paper.

    Args:
        pi (Tensor): Prediction interval bounds (0: lower, 1: higher).
        pred_area (Tensor): Area of predicted boxes.

    Returns:
        Tensor: Mean box stretch per scoring group.
    """
    assert pi.shape[0] == pred_area.shape[0], "PI and area must have same nr of boxes"
    # Assume that each scoring group of 4 coordinates is grouped together
    nr_samp, nr_scores = pi.shape[0], pi.shape[-1] // 4
    # Reshape for vectorization, (x, 8) to (x, 2, 4)
    pi_l = pi[:, 0, :].reshape(nr_samp, nr_scores, 4)
    pi_u = pi[:, 1, :].reshape(nr_samp, nr_scores, 4)
    # Compute box area based on outer PI bounds: (x1_u - x0_l) * (y1_u - y0_l)
    pi_area = (pi_u[:, :, 2] - pi_l[:, :, 0]) * (pi_u[:, :, 3] - pi_l[:, :, 1])

    return torch.repeat_interleave(
        torch.mean(torch.sqrt(pi_area / pred_area.unsqueeze(1)), dim=0), 4
    )  # dim (nr_scores)


def label_coverage(label_set: torch.Tensor, gt_class):
    """
    Returns label set coverage for a given ground truth class.
    
    Args:
        label_set (Tensor): Label set.
        gt_class (int): Ground truth class.
    
    Returns:
        Tensor: Single value tensor with label set coverage.
    """
    return label_set[:, gt_class].sum() / label_set.size(0)


def label_stratified_coverage(
    label_set: torch.Tensor,
    gt_class,
    calib_mask: torch.Tensor,
    ist: dict,
    size_bins: list = _size_bins,
    iou_bins: list = _iou_bins,
):
    """
    Returns label set coverage stratified by object area, IOU, and classification.
    
    Args:
        label_set (Tensor): Label set.
        gt_class (int): Ground truth class.
        calib_mask (Tensor): Calibration mask.
        ist (dict): Dictionary with object instance information.
        size_bins (list): List of area size bins.
        iou_bins (list): List of IOU bins.
    
    Returns:
        Tensor: Label set coverage stratified by object area, IOU, and classification.
    """
    # by area size
    area = ist["gt_area"]
    cov_area = torch.zeros((len(size_bins),))
    for i, size in enumerate(size_bins):
        mask = (~calib_mask) * (area > size[0]) * (area <= size[1])
        cov_area[i] = label_coverage(label_set[mask], gt_class)

    # by iou
    ious = ist["iou"]
    cov_iou = torch.zeros((len(iou_bins),))
    for i, iou in enumerate(iou_bins):
        mask = (~calib_mask) * (ious > iou[0]) * (ious <= iou[1])
        cov_iou[i] = label_coverage(label_set[mask], gt_class)

    # by classification
    cov_cl = torch.zeros((2,))
    if "pred_score_all" in ist.keys():
        mask_cl = (~calib_mask) * (ist["pred_score_all"].argmax(dim=1) == gt_class)
        # correctly classified
        cov_cl[0] = label_coverage(label_set[mask_cl], gt_class)
        # misclassified
        cov_cl[1] = label_coverage(label_set[~mask_cl], gt_class)
    else:
        print("Cannot compute classification-stratified metrics.")
        mask_cl = ~calib_mask

    return cov_area, cov_iou, cov_cl, mask_cl


def mean_label_set_size(label_set: torch.Tensor):
    """
    Returns mean label set size.
    
    Args:
        label_set (Tensor): Label set.
    
    Returns:
        Tensor: Single value tensor with mean label set size.
    """
    return torch.mean(label_set.sum(dim=1).float(), dim=0)


def null_set_fraction(label_set: torch.Tensor):
    """
    Returns the fraction of null label sets.
    
    Args:
        label_set (Tensor): Label set.
    
    Returns:
        Tensor: Single value tensor with null set fraction.
    """
    return torch.tensor(
        (label_set.sum(dim=1) == 0).nonzero(as_tuple=True)[0].size(0)
        / label_set.size(0)
    )


def accuracy(preds: torch.Tensor, labels: torch.Tensor):
    """
    Returns classifier accuracy.
    
    Args:
        preds (Tensor): Predicted labels.
        labels (Tensor): Ground truth labels.
    
    Returns:
        Tensor: Single value tensor with classifier accuracy.
    """
    assert preds.shape == labels.shape
    return (preds == labels).sum() / len(preds)


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    Computes the Expected Calibration Error (ECE) of a classifier.
    Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Args:
        probs (Tensor): Predicted probabilities in from (nr_samples, nr_classes).
        labels (Tensor): Ground truth labels in form (nr_samples).
        n_bins (int): Number of bins for ECE computation.
    
    Returns:
        Tensor: Single value tensor with classifier's ECE.
    """
    assert probs.shape[0] == labels.shape[0]

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    conf, preds = torch.max(probs, 1)
    acc = preds.eq(labels)

    ece = torch.tensor(0.0)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate |confidence - accuracy| in each bin
        in_bin = conf.gt(bin_lower.item()) * conf.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            acc_in_bin = acc[in_bin].float().mean()
            avg_conf_in_bin = conf[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    return ece
