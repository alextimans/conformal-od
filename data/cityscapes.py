# Copyright (c) Facebook, Inc. and its affiliates.

"""
NOTE: This file is a modified version of the original file from detectron2/data/datasets/cityscapes.py
It has been modified to map cityscape ids to coco ids and to support the conversion of cityscapes annotations to coco format.

See also some of the following references:
https://www.cityscapes-dataset.com/dataset-overview/#class-definitions
https://tillbeemelmanns.github.io/2020/10/10/convert-cityscapes-to-coco-dataset-format.html
https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion
"""

import functools
import json
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
from shapely.geometry import Polygon

from cityscapesscripts.helpers.labels import labels, id2label, name2label

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


_map_contiguous_id_to_coco_id = {
    0: 0,  # person: person
    1: 0,  # rider: person
    2: 2,  # car: car
    3: 7,  # truck: truck
    4: 5,  # bus: bus
    5: 6,  # train: train
    6: 3,  # motorcycle: motorcycle
    7: 1,  # bicycle: bicycle
}


def _get_cityscapes_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)

    # added due to error throwing
    if ".DS_Store" in cities:
        cities.remove(".DS_Store")

    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(
                city_gt_dir, basename + "gtFine_instanceIds.png"
            )
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f

    return files


def _cityscapes_files_to_dict(files, from_json, to_polygons):
    """
    Parse cityscapes annotation files to an object detection dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    image_file, instance_id_file, _, json_file = files
    annos = []

    if from_json:
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        # `polygons_union` contains the union of all valid polygons.
        polygons_union = Polygon()

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the polygon overlaps in the same way as CityscapesScripts.
        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            # Cityscapes's raw annotations uses integer coordinates
            # Therefore +0.5 here
            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
            # polygons for evaluation. This function operates in integer space
            # and draws each pixel whose center falls into the polygon.
            # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
            # We therefore dilate the input polygon by 0.5 as our input.
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                # even if we won't store the polygon it still contributes to overlaps resolution
                polygons_union = polygons_union.union(poly)
                continue

            # Take non-overlapping part of the polygon
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = label.id

            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            annos.append(anno)
    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            anno = {}
            anno["iscrowd"] = instance_id < 1000
            anno["category_id"] = label.id

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            annos.append(anno)
    ret["annotations"] = annos
    return ret


def load_cityscapes_instances(
    gt_dir, image_dir, meta_name, from_json=True, to_polygons=True
):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    files = _get_cityscapes_files(image_dir, gt_dir)

    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(
            _cityscapes_files_to_dict, from_json=from_json, to_polygons=to_polygons
        ),
        files,
    )

    # Map cityscape ids to contiguous ids
    lab = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(lab)}

    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            # map category id to contiguous id
            c_id = dataset_id_to_contiguous_id[anno["category_id"]]
            # map cityscape id to coco id
            anno["category_id"] = _map_contiguous_id_to_coco_id[c_id]

    return ret
