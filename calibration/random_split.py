import torch


def random_split(
    imgs: torch.tensor,
    ist_img_id: torch.tensor,
    calib_fraction: float,
    verbose: bool = False,
):
    """
    Identifies relevant images for the downstream task,
    randomly splits them into calibration and test sets
    and generates a boolean mask for instance-level set assignment.

    Args:
        imgs (Tensor): A boolean tensor indicating relevant images, i.e.
            images in which instances of a relevant class exist.
        ist_img_id (Tensor): Instance-level image IDs to match images
            from which instances originate from.
        calib_fraction (float): Desired fraction of calibration set for split
        verbose (bool, optional): Defaults to True.

    Returns:
        calib_mask: Boolean mask for instances in calibration set.
        calib_idx: Image indices in calibration set.
        test_idx: Image indices in test set.
    """
    # Collect all imgs where class instances occur
    img_idx = torch.nonzero(imgs, as_tuple=True)[0]
    nr_samp = len(img_idx)
    # Compute nr of calibration samples
    assert 0 <= calib_fraction <= 1, f"{calib_fraction=} not in [0,1]"
    calib_samp = torch.floor(torch.tensor(calib_fraction * nr_samp)).to(torch.int)
    # Random permutation and split
    perm = torch.randperm(nr_samp)
    calib_idx, test_idx = img_idx[perm[:calib_samp]], img_idx[perm[calib_samp:]]
    # Based on idx, create mask for respective calibration instances
    calib_mask = torch.isin(ist_img_id, torch.sort(calib_idx)[0])

    if verbose:
        print(
            f"Split {nr_samp} samples into {calib_samp} calibration and "
            f"{nr_samp - calib_samp} test samples."
        )
        print(
            f"Found {calib_mask.sum()} box instances in calibration "
            f"and {(~calib_mask).sum()} box instances in test samples.\n"
        )

    return calib_mask, calib_idx, test_idx


"""
# numpy version
# collect all imgs where class instances occur and split randomly
class_idx = 0
imgs = collector.img_list[class_idx]
img_idx = np.nonzero(imgs)[0]
perm = np.random.permutation(img_idx)
nr_calib = 20
calib_idx, test_idx = perm[:nr_calib], perm[nr_calib:]
# based on imgs, collect respective instances for downstream
d = collector.ist_list[class_idx]
calib_mask = np.isin(np.array(d["img_id"]), np.sort(calib_idx))
np.array(d["gt_x0"])[calib_mask]
"""