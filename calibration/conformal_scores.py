import torch


def abs_res(gt: torch.Tensor, pred: torch.Tensor):
    # Fixed-width absolute residual scores
    return torch.abs(gt - pred)


def norm_res(gt: torch.Tensor, pred: torch.Tensor, unc: torch.Tensor):
    # Scalable normalized residual scores
    return torch.abs(gt - pred) / unc


def quant_res(gt: torch.Tensor, pred_lower: torch.Tensor, pred_upper: torch.Tensor):
    # Scalable CQR scores, see Eq. 6 in the paper
    return torch.max(pred_lower - gt, gt - pred_upper)


def one_sided_res(gt: torch.Tensor, pred: torch.Tensor, min: bool):
    # Fixed-width one-sided scores from Andeol et al. (2023), see Eq. 6 in the paper
    return (pred - gt) if min else (gt - pred)


def one_sided_mult_res(gt: torch.Tensor, pred: torch.Tensor, mult: torch.Tensor, min: bool):
    # Scalable one-sided scores from Andeol et al. (2023), see Eq. 7 in the paper
    return (pred - gt) / mult if min else (gt - pred) / mult
