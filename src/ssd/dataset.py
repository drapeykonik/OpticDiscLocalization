import json
import os

import pandas as pd
from mmengine.fileio import dump
from PIL import Image


def convert_fundus_to_coco(ann_file, out_file, image_prefix):
    ann_frame = pd.read_csv(ann_file, sep=",", index_col=0)

    annotations = []
    images = []
    obj_count = 0
    for idx in range(len(ann_frame)):
        filename = ann_frame.loc[idx, "Image"]
        img_path = os.path.join(image_prefix, filename)
        width, height = Image.open(img_path).size
        x, y = ann_frame.loc[idx, ["X", "Y"]].values
        images.append(
            dict(id=idx, file_name=filename, height=height, width=width)
        )

        data_anno = dict(
            image_id=idx,
            id=obj_count,
            category_id=0,
            bbox=[x - 250, y - 250, 500, 500],
            area=500 * 500,
            segmentation=None,
            iscrowd=0,
        )
        annotations.append(data_anno)
        obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{"id": 0, "name": "optic_disc"}],
    )
    dump(coco_format_json, out_file)


if __name__ == "__main__":
    convert_fundus_to_coco(
        "data/processed/localization/train/location.csv",
        "data/processed/localization/train/annotation_coco.json",
        "data/processed/localization/train",
    )
    convert_fundus_to_coco(
        "data/processed/localization/valid/location.csv",
        "data/processed/localization/valid/annotation_coco.json",
        "data/processed/localization/valid",
    )
    convert_fundus_to_coco(
        "data/processed/localization/test/location.csv",
        "data/processed/localization/test/annotation_coco.json",
        "data/processed/localization/test",
    )
