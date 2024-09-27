import sys
sys.path.insert(0, "/home/atimans/Desktop/project_1/conformalbb/detectron2")

import torch
from torch import Tensor

from detectron2.structures.boxes import Boxes, matched_pairwise_iou
from util.io_file import save_json


_default_dict_fields = [
    "gt_x0",
    "gt_y0",
    "gt_x1",
    "gt_y1",
    "pred_x0",
    "pred_y0",
    "pred_x1",
    "pred_y1",
    "gt_centers",
    "pred_centers",
    "gt_area",
    "pred_area",
    "pred_score",
    "pred_score_all",
    "pred_logits_all",
    "iou",
    "img_id",
    "label_score",
]


class DataCollector:
    """
    Collects relevant data e.g. bounding box coordinates for downstream tasks
    for a given image with corresponding instance data. The data is collected
    in objects serializable with JSON. The motivation is to permit precomputing
    all relevant inference results, such that running the conformal procedures 
    for multiple trials is highly efficient. Simlarly, data storage is efficient.
    It comes at the cost of somewhat verbose and complex data structures. 
    

    Key objects collecting information are:
    - img_list: A list of lists, each index is equivalent to one class,
        each element is a list of booleans for all images indicating which
        images contain box instances of that specific class.
    - ist_list: A list of dicts, each index is equivalent to one class,
        each element is a dict containing instance-wise and coordinate-wise
        information for all class-relevant instances.
    """

    def __init__(
        self,
        nr_class: int,
        nr_img: int,
        dict_fields: list = _default_dict_fields,
        logger=None,
        label_set_generator=None,
    ):
        self.nr_class = nr_class
        self.nr_img = nr_img
        self.dict_fields = dict_fields
        self.img_list = [
            torch.zeros(self.nr_img).tolist() for _ in range(self.nr_class)
        ]
        self.ist_list = [
            {k: [] for k in self.dict_fields} for _ in range(self.nr_class)
        ]
        logger.info(  # type:ignore
            f"Initialized img_list and ist_list for {self.nr_class} classes "
            f"and {self.nr_img} images, with info for the following "
            f"fields: \n{self.dict_fields}."
        )
        self.label_set_generator = label_set_generator

    def __call__(
        self,
        gt_box: Boxes,
        gt_class: Tensor,
        pred_box: Boxes,
        pred_score: Tensor,
        pred_score_all: Tensor,
        pred_logits_all: Tensor,
        img_id: int,
        verbose: bool = False,
    ):
        for c in torch.unique(gt_class).numpy():
            # img has instances of class
            self.img_list[c][img_id] = 1
            # indices for matching instances
            idx = torch.nonzero(gt_class == c, as_tuple=True)[0]
            # add info for instances
            self._add_instances(
                c,
                img_id,
                idx,
                gt_box,
                pred_box,
                pred_score,
                pred_score_all,
                pred_logits_all,
            )

        if verbose:
            print(f"Added all instances for image {img_id}.")

    def _add_instances(
        self,
        c: int,
        img_id: int,
        idx: Tensor,
        gt_box: Boxes,
        pred_box: Boxes,
        pred_score: Tensor,
        pred_score_all: Tensor,
        pred_logits_all: Tensor,
    ):
        # Boxes coord come as BoxMode.XYXY_ABS: (x0, y0, x1, y1)
        self.ist_list[c]["gt_x0"] += gt_box[idx].tensor[:, 0].tolist()
        self.ist_list[c]["gt_y0"] += gt_box[idx].tensor[:, 1].tolist()
        self.ist_list[c]["gt_x1"] += gt_box[idx].tensor[:, 2].tolist()
        self.ist_list[c]["gt_y1"] += gt_box[idx].tensor[:, 3].tolist()

        self.ist_list[c]["pred_x0"] += pred_box[idx].tensor[:, 0].tolist()
        self.ist_list[c]["pred_y0"] += pred_box[idx].tensor[:, 1].tolist()
        self.ist_list[c]["pred_x1"] += pred_box[idx].tensor[:, 2].tolist()
        self.ist_list[c]["pred_y1"] += pred_box[idx].tensor[:, 3].tolist()

        self.ist_list[c]["gt_centers"] += gt_box[idx].get_centers().tolist()
        self.ist_list[c]["pred_centers"] += pred_box[idx].get_centers().tolist()
        self.ist_list[c]["gt_area"] += gt_box[idx].area().tolist()
        self.ist_list[c]["pred_area"] += pred_box[idx].area().tolist()

        self.ist_list[c]["pred_score"] += pred_score[idx].tolist()
        if pred_score_all is not None:
            self.ist_list[c]["pred_score_all"] += pred_score_all[idx].tolist()
        if pred_logits_all is not None:
            self.ist_list[c]["pred_logits_all"] += pred_logits_all[idx].tolist()

        self.ist_list[c]["iou"] += matched_pairwise_iou(
            gt_box[idx], pred_box[idx]
        ).tolist()
        self.ist_list[c]["img_id"] += [img_id for _ in range(len(idx))]
        if self.label_set_generator is not None:
            self.ist_list[c]["label_score"] += self.label_set_generator.score(
                pred_score_all[idx], c
            ).tolist()

    def to_file(self, filename: str, filedir: str, **kwargs):
        save_json(self.img_list, filename + "_img_list", filedir, **kwargs)
        save_json(self.ist_list, filename + "_ist_list", filedir, **kwargs)
