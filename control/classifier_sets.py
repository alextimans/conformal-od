import torch
from tqdm import tqdm

from calibration import pred_intervals
from evaluation import metrics


class LabelSet:
    """
    This class is a base class for label set generation strategies.
    It provides the basic structure for label set generation and
    evaluation of label set-based metrics.
    
    In particular, it provides the following methods:
    - collect: Initializes the necessary data for label set generation.
    - get_pred_set: Generates the label set based on the prediction scores.
    - handle_null_set: Handles empty label sets by replacing them with a singleton set.
    - __call__: Main method for label set generation and evaluation.
    - compute_box_metrics: Computes box set metrics based on the label set.
    
    Note that somewhat unelegantly (and with room for better modularization),
    once the label sets are generated and the box quantils are selected on that basis,
    the box set metrics are recomputed in compute_box_metrics by retrieving the bounding box predictions,
    building renewed prediction intervals, and computing the box set metrics.
    """
    def __init__(self, cfg, args, logger):
        self.logger = logger
        self.risk_control = args.risk_control
        self.label_set = args.label_set

        self.seed = cfg.PROJECT.SEED
        self.calib_trials = cfg.CALIBRATION.TRIALS
        self.box_set_strategy = cfg.CALIBRATION.BOX_SET_STRATEGY
        self.label_alpha = args.label_alpha
        self.save_label_set = args.save_label_set

        self.one_sided_pi = True if "ONE_SIDED_PI" in cfg.CALIBRATION.keys() else False
        if "QUANTILE_REGRESSION" in cfg.keys():
            q_str = [f"pred_q{int(q*100)}_" for q in cfg.QUANTILE_REGRESSION.QUANTILES]
            self.cqr_lower, self.cqr_upper = (
                q_str[i] for i in cfg.QUANTILE_REGRESSION.QUANTILE_INDICES
            )  # equates to "pred_q5_", "pred_q95_"

    def collect(
        self,
        img_list,
        ist_list,
        nr_class,
        nr_metrics,
        nr_scores,
        score_fields,
        coord_fields,
    ):
        self.img_list = img_list
        self.ist_list = ist_list
        self.nr_class = nr_class
        self.nr_metrics = nr_metrics + 4  # 4 slots for strat by misclassif.
        self.nr_label_metrics = 15
        self.nr_scores = nr_scores
        self.score_fields = score_fields
        self.coord_fields = coord_fields

        # create additional fields for label set
        self.calib_masks = [
            [torch.tensor(0) for _ in range(self.nr_class)]
            for _ in range(self.calib_trials)
        ]
        self.box_quantiles = torch.zeros(
            size=(self.calib_trials, self.nr_class, self.nr_scores)
        )
        self.label_quantiles = torch.zeros(size=(self.calib_trials, self.nr_class))
        self.nr_ists = [
            len(self.ist_list[c]["img_id"]) for c in range(0, self.nr_class)
        ]

    def get_pred_set(self, pred_score_all):
        # overriden in each subclass
        return torch.tensor([])

    def handle_null_set(self, pred_score_all: torch.Tensor, label_sets: torch.Tensor):
        # replace empty sets with singleton set of highest probability class
        indices = (label_sets.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if indices.numel() != 0:
            top_label = pred_score_all[indices].argmax(dim=1)
            label_sets[indices, top_label] = 1
            # label_sets[indices, :] = 1  # alternative: replace with full set
        return label_sets

    def __call__(self):
        self.logger.info(
            f"""
            Running label set procedure for {self.calib_trials} trials...
            Label set strategy: {self.label_set} / {self.__class__.__name__},
            box set strategy: {self.box_set_strategy}, label set alpha: {self.label_alpha}.
            """
        )

        if self.save_label_set:
            label_sets = torch.zeros(
                size=(self.calib_trials, sum(self.nr_ists), self.nr_class)
            ).to(torch.bool)
        else:
            label_sets = torch.tensor([])

        label_data = torch.zeros(
            size=(self.calib_trials, self.nr_class, self.nr_label_metrics)
        )
        box_set_data = torch.zeros(
            size=(self.calib_trials, self.nr_class, self.nr_scores, self.nr_metrics)
        )

        for t in tqdm(range(self.calib_trials), desc="Trials"):
            for c in range(self.nr_class):
                ist = {k: torch.tensor(v) for k, v in self.ist_list[c].items()}
                calib_mask = self.calib_masks[t][c]
                self.label_q = self.label_quantiles[t]
                self.nr_ist = self.nr_ists[c]
                nr_ist_upto = sum(self.nr_ists[:c])

                if self.nr_ist == 0:  # no ist for this class
                    continue

                # compute label sets; binary (nr_ist, nr_class) tensor
                label_set = self.get_pred_set(ist["pred_score_all"])
                null_set_frac = metrics.null_set_fraction(label_set[~calib_mask])
                label_set = self.handle_null_set(ist["pred_score_all"], label_set)

                if self.save_label_set:
                    label_sets[
                        t, nr_ist_upto : (nr_ist_upto + self.nr_ist), :
                    ] = label_set.to(torch.bool)

                # compute label set metrics
                nr_calib_samp = calib_mask.sum()
                mean_set_size = metrics.mean_label_set_size(label_set[~calib_mask])
                cov_set = metrics.label_coverage(label_set[~calib_mask], c)
                (
                    cov_set_area,
                    cov_set_iou,
                    cov_set_cl,
                    mask_cl,
                ) = metrics.label_stratified_coverage(label_set, c, calib_mask, ist)
                mean_set_size_cl = metrics.mean_label_set_size(label_set[mask_cl])
                mean_set_size_miscl = metrics.mean_label_set_size(label_set[~mask_cl])

                label_data[t, c, :] = torch.cat(
                    (
                        nr_calib_samp.ravel(),
                        self.label_q[c].ravel(),
                        null_set_frac.ravel(),
                        mean_set_size.ravel(),
                        cov_set.ravel(),
                        cov_set_area,
                        cov_set_iou,
                        cov_set_cl,
                        mean_set_size_cl.ravel(),
                        mean_set_size_miscl.ravel(),
                    ),
                    dim=0,
                )

                # label set-based box quantile selection strategy
                box_set_quant, _ = box_set_strategy(
                    label_set,
                    self.box_quantiles[t],
                    self.box_set_strategy,
                )

                # risk-control specific box set metrics calculations
                box_set_data[t, c, :, :] = self.compute_box_metrics(
                    ist, calib_mask, mask_cl, box_set_quant
                )

                del calib_mask

        return label_sets, label_data, box_set_data

    def compute_box_metrics(
        self,
        ist: dict,
        calib_mask: torch.Tensor,
        mask_cl: torch.Tensor,
        box_set_quant: torch.Tensor,
    ):
        gt = torch.zeros(size=(self.nr_ist, self.nr_scores), dtype=torch.float32)

        if self.risk_control == "std_conf":
            pred = torch.zeros_like(gt)
            for i, s in enumerate(self.score_fields):
                gt[:, i] = ist["gt_" + self.coord_fields[i % 4]]
                pred[:, i] = ist["pred_" + self.coord_fields[i % 4]]
            pi = pred_intervals.fixed_pi(pred, box_set_quant)

        elif self.risk_control == "ens_conf":
            pred = torch.zeros_like(gt)
            unc = torch.zeros_like(gt)
            for i, s in enumerate(self.score_fields):
                gt[:, i] = ist["gt_" + self.coord_fields[i % 4]]
                pred[:, i] = ist["pred_" + self.coord_fields[i % 4]]
                unc[:, i] = ist["unc_" + self.coord_fields[i % 4]]
            if self.one_sided_pi:
                pi = pred_intervals.one_sided_pi_box_set(
                    pred, unc, box_set_quant, ist["pred_centers"]
                )
            else:
                pi = pred_intervals.norm_pi(pred, unc, box_set_quant)

        elif self.risk_control == "cqr_conf":
            pred_lower = torch.zeros_like(gt)
            pred_upper = torch.zeros_like(gt)
            for i, s in enumerate(self.score_fields):
                gt[:, i] = ist["gt_" + self.coord_fields[i % 4]]
                pred_lower[:, i] = ist[self.cqr_lower + self.coord_fields[i % 4]]
                pred_upper[:, i] = ist[self.cqr_upper + self.coord_fields[i % 4]]
            pi = pred_intervals.quant_pi(pred_lower, pred_upper, box_set_quant)

        elif self.risk_control == "base_conf":
            pred = torch.zeros_like(gt)
            for i, s in enumerate(self.score_fields):
                gt[:, i] = ist["gt_" + self.coord_fields[i % 4]]
                pred[:, i] = ist["pred_" + self.coord_fields[i % 4]]
            # Scaling factors for one sided PIs
            scale_add = torch.ones((pred.shape[0], 4))
            scale_mult = torch.stack(
                (
                    (pred[:, 2] - pred[:, 0]),
                    (pred[:, 3] - pred[:, 1]),
                    (pred[:, 2] - pred[:, 0]),
                    (pred[:, 3] - pred[:, 1]),
                ),
                dim=1,
            )
            # Approach adapted to create two-sided PIs by considering box centers
            pi = torch.cat(
                (
                    pred_intervals.one_sided_pi_box_set(  # for one_sided_res
                        pred[:, :4],
                        scale_add,
                        box_set_quant[:, :4],
                        ist["pred_centers"],
                    ),
                    pred_intervals.one_sided_pi_box_set(  # for one_sided_mult_res
                        pred[:, 4:],
                        scale_mult,
                        box_set_quant[:, 4:],
                        ist["pred_centers"],
                    ),
                ),
                dim=-1,
            )

        else:
            raise ValueError("Risk control not specified.")

        nr_calib_samp = calib_mask.sum().repeat(self.nr_scores)
        cov_coord, cov_box = metrics.coverage(gt[~calib_mask], pi[~calib_mask])
        cov_area, cov_iou = metrics.stratified_coverage(gt, pi, calib_mask, ist)
        mpiw = metrics.mean_pi_width(pi[~calib_mask])
        stretch = metrics.box_stretch(pi[~calib_mask], ist["pred_area"][~calib_mask])

        _, cov_box_cl = metrics.coverage(gt[mask_cl], pi[mask_cl])
        _, cov_box_miscl = metrics.coverage(gt[~mask_cl], pi[~mask_cl])
        mpiw_cl = metrics.mean_pi_width(pi[mask_cl])
        mpiw_miscl = metrics.mean_pi_width(pi[~mask_cl])

        metr = torch.zeros((self.nr_scores, self.nr_metrics))
        metr[:, :6] = torch.stack(
            (
                nr_calib_samp,
                box_set_quant.mean(dim=0),
                mpiw,
                stretch,
                cov_coord,
                cov_box,
            ),
            dim=1,
        )
        metr[:, 6:9] = cov_area
        metr[:, 9:12] = cov_iou
        metr[:, 12:] = torch.stack(
            (cov_box_cl, cov_box_miscl, mpiw_cl, mpiw_miscl), dim=1
        )

        return metr


class TopSingletonSet(LabelSet):
    """
    Singleton label set (Top), i.e. return highest probability class every time.
    """

    def score(self, pred_score_all: torch.Tensor, gt_class):
        # irrelevant, so return arbitrary value
        return torch.ones_like(pred_score_all[:, gt_class])

    def get_pred_set(self, pred_score_all: torch.Tensor):
        # return singleton set of highest probability class
        label_sets = torch.zeros_like(pred_score_all)
        label_sets[torch.arange(len(pred_score_all)), pred_score_all.argmax(dim=1)] = 1
        return label_sets


class FullSet(LabelSet):
    """
    Full label set (Full), i.e. include all classes every time.
    """

    def score(self, pred_score_all: torch.Tensor, gt_class):
        # irrelevant, so return arbitrary value
        return torch.ones_like(pred_score_all[:, gt_class])

    def get_pred_set(self, pred_score_all: torch.Tensor):
        return torch.ones_like(pred_score_all)


class OracleSet(LabelSet):
    """
    Assuming the prediction of correct conditional probabilities P(y|x), i.e.
    perfect calibration, we return the 'optimal' density level sets.
    That is, we order P(y|x) by descending probability and return the
    density level set such that the sum of probabilities is at least 1-alpha.
    
    NOTE: This approach was renamed to 'Naive' in the paper, since it is not
    a true oracle in the sense that there is still an empirical quantile selection
    procedure, but it is 'naive' in the sense that it assumes perfect calibration
    to ensure nominal coverage.
    """

    def score(self, pred_score_all: torch.Tensor, gt_class):
        # irrelevant for density level sets, so return arbitrary value
        return torch.ones_like(pred_score_all[:, gt_class])

    def get_pred_set(self, pred_score_all: torch.Tensor):
        # add jitter for non-ambiguous sorting
        eps = 1e-6
        pred_score_all += torch.rand_like(pred_score_all) * eps - eps / 2
        # renormalize scores to sum to 1, since background class is not included
        pred_score_all /= pred_score_all.sum(dim=1, keepdim=True)
        # get density level set mask information
        sorted_pred_score_all, idx = torch.sort(pred_score_all, dim=1, descending=True)
        thresh_mask = torch.cumsum(sorted_pred_score_all, dim=1) >= (
            1 - self.label_alpha
        )
        # get density level set index boolean mask
        idx_mask = torch.zeros_like(pred_score_all)
        idx_mask[~thresh_mask] = 1
        idx_mask[torch.arange(len(thresh_mask)), thresh_mask.int().argmax(dim=1)] = 1
        # get label set based on density level set mask
        label_sets = torch.zeros_like(pred_score_all)
        label_sets[idx_mask.nonzero(as_tuple=True)[0], idx[idx_mask.bool()]] = 1
        return label_sets


class ClassThresholdSet(LabelSet):
    """
    Label sets via conformal class-conditional thresholding (ClassThr), i.e.
    LABEL set classifier from (Sadinle et al. 2019).
    Scores are (1-prob) instead of (prob) to better reflect
    nonconformity interpretation, see e.g. (Angelopoulos & Bates 2023).
    
    This is our preferred label set strategy advocated in the paper (ClassThr).
    """

    def score(self, pred_score_all: torch.Tensor, gt_class):
        # 1 - class probability of ground truth class
        return 1 - pred_score_all[:, gt_class]

    def get_pred_set(self, pred_score_all: torch.Tensor, q=None):
        label_q = self.label_q if q is None else q
        # get label sets via (class-conditional) thresholding
        return (pred_score_all >= 1 - label_q).int()


def get_label_set_generator(cfg, args, logger):
    # instantiaties class based on desired label set strategy
    label_set = args.label_set
    logger.info(f"Instantiating label set generator '{label_set}'.")

    if label_set == "top_singleton":
        return TopSingletonSet(cfg, args, logger)
    elif label_set == "full":
        return FullSet(cfg, args, logger)
    elif label_set == "oracle":
        return OracleSet(cfg, args, logger)
    elif label_set == "class_threshold":
        return ClassThresholdSet(cfg, args, logger)
    else:
        raise ValueError("Label set not specified.")


def box_set_strategy(
    label_set: torch.Tensor, box_quantiles: torch.Tensor, box_set_strategy: str
):
    """
    This function selects the box quantiles based on the label set.
    In the paper (and implemented here) we only consider the 'max' strategy,
    i.e. selecting the maximum box quantiles for all classes in the label set.
    """
    quant = torch.zeros((label_set.size(0), box_quantiles.size(1)), dtype=torch.float32)
    quant_idx = torch.zeros_like(quant, dtype=torch.int32)

    if box_set_strategy == "max":
        # max over label set box quantiles per coordinate for each instance
        for i, label_mask in enumerate(label_set):
            quant[i, :], quant_idx[i, :] = box_quantiles[label_mask.bool()].max(dim=0)
    else:
        raise ValueError("Box set strategy not specified.")

    return quant, quant_idx
