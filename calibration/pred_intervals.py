import torch
# For copula baselines uncomment (see https://arxiv.org/abs/2311.10900):
#   import numpy as np
#   from copulae import GumbelCopula, EmpiricalCopula, pseudo_obs


def get_quantile(scores: torch.Tensor, alpha: torch.Tensor, n, verbose: bool = False):
    """
    Get conformal quantile from calibration samples.
    Conformal quantile formula: ceil[(1-alpha)(n+1)]/n

    Args:
        scores (Tensor): Conformity scores computed for calibration samples.
        alpha (Tensor): Desired nominal coverage levels (per coordinate).
        n (Tensor): Nr. of calibration samples

    Returns:
        Tensor: Alpha-level quantile of conformity scores.
    """
    q = (torch.ceil((1 - alpha) * (n + 1)) / n).clamp(0, 1)
    if verbose:
        print(f"Nominal quantiles:{1 - alpha}")
        print(f"Sample-corrected quantiles: {q}")
    return torch.quantile(scores, q, dim=0, interpolation="higher")


def compute_quantile(
    scores: torch.Tensor, box_correction: str, nr_samp=None, alpha=0.1
):
    """
    Compute quantile for the desired nominal coverage level with 
    different box correction methods (multiple testing corrections).
    
    Args:
        scores (Tensor): Conformity scores computed for calibration samples.
        box_correction (str): Type of box correction to apply.
        nr_samp (int, optional): Number of calibration samples. Defaults to None.
        alpha (float, optional): Desired nominal coverage level. Defaults to 0.1.
    
    Returns:
        Tensor: Alpha-level multiple testing-corrected quantile of conformity scores.
    """
    assert 0 <= alpha <= 1, f"Nominal coverage {alpha=} not in [0,1]"
    alpha = torch.tensor([alpha])
    n = torch.tensor(nr_samp if nr_samp is not None else scores.shape[0])
    nr_scores = scores.shape[1]
    quant = torch.empty(size=(nr_scores,), dtype=torch.float32)

    if box_correction == "naive_max":
        # max over coordinate quantiles as global quantile 
        q = get_quantile(scores, alpha, n)
        for s in range(nr_scores // 4):
            i, j = s * 4, s * 4 + 4
            quant[i:j] = torch.max(q[:, i:j])

    elif box_correction == "bonferroni":
        # FWER Bonferroni correction
        alpha_bonf = alpha / 4
        quant[:] = get_quantile(scores, alpha_bonf, n)

    elif box_correction == "bonferroni_sidak":
        # FWER Bonferroni correction with Sidak improvement
        alpha_bsidak = 1 - (1 - alpha) ** 0.25
        quant[:] = get_quantile(scores, alpha_bsidak, n)

    elif box_correction == "rank_global":
        # Max-rank algorithm v1: coordinate quantiles as informed via global quantile over max ranks
        ranks = torch.argsort(torch.argsort(scores, dim=0), dim=0)
        for s in range(nr_scores // 4):
            i, j = s * 4, s * 4 + 4
            max_rank = torch.max(ranks[:, i:j], dim=1)[0]
            q_rank = get_quantile(max_rank.to(torch.float32), alpha, n)
            quant[i:j] = torch.sort(scores, dim=0)[0][int(q_rank), i:j]

    elif box_correction == "rank_coord":
        # Max-rank algorithm v2: coordinate quantiles as informed via global quantile over max ranks with coordinate-wise improvement
        # This method usually gives slightly tighter intervals than Max-rank algorithm v1,
        # but is also prone to slight undercoverage (within empirical target range)
        ranks = torch.argsort(torch.argsort(scores, dim=0), dim=0)
        for s in range(nr_scores // 4):
            i, j = s * 4, s * 4 + 4
            argsort_max_rank = torch.argsort(torch.max(ranks[:, i:j], dim=1)[0])
            q_rank = get_quantile(argsort_max_rank.to(torch.float32), alpha, n)
            incl_el = argsort_max_rank[: int(q_rank)]
            if incl_el.numel() == 0:  # incl_el is empty (degenerate case)
                incl_el = torch.tensor([0])
            q_rank_coord = torch.max(ranks[incl_el, i:j], dim=0)[0]
            quant[i:j] = torch.diagonal(torch.sort(scores, dim=0)[0][q_rank_coord, i:j])

    elif box_correction == "score_global":
        # Global quantile using max on scores directly instead of on ranks (as in Max-rank v1)
        # This correction is also considered by Andeol et al. (2023), 
        # and in fact equates to Westfall & Young's 'max-T' correction.
        for s in range(nr_scores // 4):
            i, j = s * 4, s * 4 + 4
            max_score = torch.max(scores[:, i:j], dim=1)[0]
            quant[i:j] = get_quantile(max_score, alpha, n)
    
    # For copula baselines uncomment (see https://arxiv.org/abs/2311.10900):
    
    # elif box_correction == "gumbel_copula":
    #     cop = GumbelCopula(dim=4)
    #     for s in range(nr_scores // 4):
    #         i, j = s * 4, s * 4 + 4
    #         if (scores[:, i:j].shape[0] < 4):
    #             alpha_gumb = alpha
    #         else:
    #             cop.fit(scores[:, i:j].numpy())
    #             theta = cop.params
    #             alpha_gumb = 1 - (1 - alpha)**(1/(4**(1/theta)))
    #         quant[i:j] = get_quantile(scores[:, i:j], alpha_gumb, n)

    # elif box_correction == "emp_copula":
    #     # No analytical solution so create alpha search space (0, alpha+0.1)
    #     test_alpha = np.tile(np.arange(0.0, alpha+0.1, step=1e-4), (4, 1)).T
    #     for s in range(nr_scores // 4):
    #         i, j = s * 4, s * 4 + 4
    #         ecop = EmpiricalCopula(pseudo_obs(scores[:, i:j].numpy()))
    #         alpha_emp = test_alpha[ecop.cdf(1-test_alpha) >= (1-alpha.item())].max()
    #         quant[i:j] = get_quantile(scores[:, i:j], alpha_emp, n)

    else:  # no box correction
        quant[:] = get_quantile(scores, alpha, n)

    return quant


def fixed_pi(pred: torch.Tensor, quantile: torch.Tensor):
    return torch.stack((pred - quantile, pred + quantile), dim=1)


def norm_pi(pred: torch.Tensor, unc: torch.Tensor, quantile: torch.Tensor):
    return torch.stack((pred - unc * quantile, pred + unc * quantile), dim=1)


def quant_pi(pred_lower: torch.Tensor, pred_upper: torch.Tensor, quantile: torch.Tensor):
    # CQR prediction interval, see Eq. 7 in the paper
    return torch.stack((pred_lower - quantile, pred_upper + quantile), dim=1)


def one_sided_pi(pred: torch.Tensor, scale: torch.Tensor, quantile: torch.Tensor, centers: torch.Tensor):
    # Interval construction following Andeol et al. (2023), see Eqs 12-15 in the paper
    # To create a two-sided interval, we assign the other respective side to the prediction box center.
    return torch.stack(
        (
            torch.stack(
                (pred[:, 0] - scale[:, 0] * quantile[0], centers[:, 0]), dim=-1
            ),
            torch.stack(
                (pred[:, 1] - scale[:, 1] * quantile[1], centers[:, 1]), dim=-1
            ),
            torch.stack(
                (centers[:, 0], pred[:, 2] + scale[:, 2] * quantile[2]), dim=-1
            ),
            torch.stack(
                (centers[:, 1], pred[:, 3] + scale[:, 3] * quantile[3]), dim=-1
            ),
        ),
        dim=-1,
    )


def one_sided_pi_box_set(pred: torch.Tensor, scale: torch.Tensor, quantile: torch.Tensor, centers: torch.Tensor):
    # Interval construction following Andeol et al. (2023), see Eqs 12-15 in the paper
    # To create a two-sided interval, we assign the other respective side to the prediction box center.
    # This function is used when the label sets are also formed, since the shape of the quantile tensor is different.
    return torch.stack(
        (
            torch.stack(
                (pred[:, 0] - scale[:, 0] * quantile[:, 0], centers[:, 0]), dim=-1
            ),
            torch.stack(
                (pred[:, 1] - scale[:, 1] * quantile[:, 1], centers[:, 1]), dim=-1
            ),
            torch.stack(
                (centers[:, 0], pred[:, 2] + scale[:, 2] * quantile[:, 2]), dim=-1
            ),
            torch.stack(
                (centers[:, 1], pred[:, 3] + scale[:, 3] * quantile[:, 3]), dim=-1
            ),
        ),
        dim=-1,
    )
