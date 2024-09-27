"""
This file contains the QuantileROIHead and QuantileFastRCNNOutputLayer classes,
which are used for training and inference of the object detector model with added
quantile regression predictions, used for the CQR conformal procedure.
"""

from typing import Tuple, Union, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    FastRCNNOutputLayers,
)
from detectron2.modeling.box_regression import (
    Box2BoxTransform,
    _dense_box_regression_loss,
)
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.layers import cat, cross_entropy, nonzero_tuple, batched_nms
from detectron2.utils.events import get_event_storage

from util import io_file


@ROI_HEADS_REGISTRY.register()
class QuantileROIHead(StandardROIHeads):
    """
    A ROI head used for quantile regression.
    Same as StandardROIHeads, but with a different output layer.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.box_predictor = QuantileFastRCNNOutputLayer(
            cfg, self.box_head.output_shape
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        instances, losses = super().forward(images, features, proposals, targets)
        return instances, losses


class QuantileFastRCNNOutputLayer(FastRCNNOutputLayers):
    """
    A quantile regression output layer,
    modelled after and modified from FastRCNNOutputLayers.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        # add quantile heads
        cfg_q = io_file.load_yaml("qr_cfg", "conformalbb/model", True)
        self.quantiles = cfg_q.QUANTILES
        self.q_names = [f"bbox_pred_q{int(q*100)}" for q in self.quantiles]
        q_in = self.bbox_pred.in_features
        q_out = self.bbox_pred.out_features
        for q_name in self.q_names:
            setattr(self, q_name, nn.Linear(q_in, q_out))
            nn.init.normal_(getattr(self, q_name).weight, std=0.001)
            nn.init.constant_(getattr(self, q_name).bias, 0)

        # self.box_reg_loss_type = ...

    def forward(self, x):
        """
        see super.forward
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        proposal_deltas_q = [getattr(self, q_name)(x) for q_name in self.q_names]

        return scores, proposal_deltas, proposal_deltas_q

    def losses(self, predictions, proposals):
        """
        see super.losses
        """
        scores, proposal_deltas, proposal_deltas_q = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0
            )  # Nx4
            assert (
                not proposal_boxes.requires_grad
            ), "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [
                    (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                    for p in proposals
                ],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device
            )

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        # modified box regression loss to include quantile losses
        loss_box_reg = self.box_reg_loss(
            proposal_boxes, gt_boxes, proposal_deltas, proposal_deltas_q, gt_classes
        )
        losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(
        self, proposal_boxes, gt_boxes, pred_deltas, pred_deltas_q, gt_classes
    ):
        """
        see super.box_reg_loss
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
            fg_pred_deltas_q = [p[fg_inds] for p in pred_deltas_q]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]
            fg_pred_deltas_q = [
                p.view(-1, self.num_classes, box_dim)[fg_inds, gt_classes[fg_inds]]
                for p in pred_deltas_q
            ]

        # mean regression loss
        loss_box_reg_m = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )
        # quantile losses
        loss_box_reg_q = []
        assert len(fg_pred_deltas_q) == len(self.quantiles)
        for i, p in enumerate(fg_pred_deltas_q):
            loss_box_reg_q.append(
                self.quantile_box_regression_loss(
                    torch.tensor(self.quantiles[i]),
                    [proposal_boxes[fg_inds]],
                    self.box2box_transform,
                    [p.unsqueeze(0)],
                    [gt_boxes[fg_inds]],
                    ...,
                )
            )
        # transform/aggregate quantile losses
        loss_box_reg_q = torch.stack(loss_box_reg_q, dim=0).sum()
        # aggregate with regression loss
        # loss_box_reg = loss_box_reg_m + loss_box_reg_q
        loss_box_reg = loss_box_reg_q  # since mean pred head is frozen

        return loss_box_reg / max(gt_classes.numel(), 1.0)

    def quantile_box_regression_loss(
        self,
        quant: torch.Tensor,
        anchors: List[Union[Boxes, torch.Tensor]],
        box2box_transform: Box2BoxTransform,
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        fg_mask: torch.Tensor,
    ):
        """
        partially see box_regression._dense_box_regression_loss
        """
        if isinstance(anchors[0], Boxes):
            anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        else:
            anchors = cat(anchors)
        # following transform for smooth_l1_loss
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        # compute quantile loss
        loss_box_reg = quantile_loss(
            pred=cat(pred_anchor_deltas, dim=1)[fg_mask],
            target=gt_anchor_deltas[fg_mask],
            quantile=quant,
        )

        return loss_box_reg

    def inference(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
        proposals: List[Instances],
    ):
        """
        see super.inference
        """
        boxes, boxes_q = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            boxes_q,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.quantiles,
        )

    def predict_boxes(self, predictions, proposals):
        """
        see super.predict_boxes
        """
        if not len(proposals):
            return []
        _, proposal_deltas, proposal_deltas_q = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        predict_boxes_q = [
            self.box2box_transform.apply_deltas(prop_delta_q, proposal_boxes)
            for prop_delta_q in proposal_deltas_q
        ]
        return predict_boxes.split(num_prop_per_image), [
            list(pred_box_q.split(num_prop_per_image)) for pred_box_q in predict_boxes_q
        ]

    def predict_probs(self, predictions, proposals):
        """
        see super.predict_probs
        """
        scores, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    quantile or pinball loss
    modelled after https://github.com/yromano/cqr/blob/master/cqr/torch_models.py
    reduction="sum" following smooth_l1_loss
    """
    assert pred.shape == target.shape, "input and target must have the same shape"
    assert 0 < quantile < 1, "quantile must be in (0, 1)"

    err = target - pred
    loss = torch.max((quantile - 1.0) * err, quantile * err)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    boxes_q: List,
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    quantiles: List[float],
):
    """
    see fast_rcnn.fast_rcnn_inference()
    """
    # From [ [(Ri_q1, K*4), ...], [(Ri_q2, K*4), ...], ... ], i.e. list of image lists per quantile
    # to [ [(Ri_q1, K*4), (Ri_q2, K*4), ...], ... ], i.e. list of quantile lists per image
    boxes_q_by_image = [list(box_per_q) for box_per_q in zip(*boxes_q)]

    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            boxes_q_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            quantiles,
        )
        for boxes_per_image, boxes_q_per_image, scores_per_image, image_shape in zip(
            boxes, boxes_q_by_image, scores, image_shapes
        )
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    boxes_q,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    quantiles: List[float],
):
    """
    see fast_rcnn.fast_rcnn_inference_single_image()
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        boxes_q = [box_q[valid_mask] for box_q in boxes_q]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4

    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    boxes_q = [Boxes(box_q.reshape(-1, 4)) for box_q in boxes_q]
    for box_q in boxes_q:
        box_q.clip(image_shape)
    boxes_q = [box_q.tensor.view(-1, num_bbox_reg_classes, 4) for box_q in boxes_q]

    # MODIFIED: save full score vector
    scores_all = scores.clone()

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        boxes_q = [box_q[filter_inds[:, 0], 0] for box_q in boxes_q]
    else:
        boxes = boxes[filter_mask]
        boxes_q = [box_q[filter_mask] for box_q in boxes_q]
    scores = scores[filter_mask]

    # MODIFIED: filter full score vector
    scores_all = scores_all[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    # Run NMS only on boxes and let indices inform boxes_q
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    boxes_q = [box_q[keep] for box_q in boxes_q]

    # MODIFIED: index full score vector
    scores_all = scores_all[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    for q, box_q in zip(quantiles, boxes_q):
        result.set(f"pred_boxes_q{int(q*100)}", Boxes(box_q))
    # MODIFIED: add full score vector
    result.scores_all = scores_all
    return result, filter_inds[:, 0]


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)
