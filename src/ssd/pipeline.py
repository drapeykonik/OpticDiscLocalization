import json
import os
from typing import List, Tuple

import numpy as np
import torch
from mmcv import Compose
from mmdet import apis
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.runner import Runner
from PIL import Image, ImageDraw
from torch.nn.functional import mse_loss

from src.metrics import localization_accuracy


class Pipeline:
    def __init__(self, config: str, log_dir: os.PathLike = None) -> None:
        self.config_path = config
        self.log_dir = log_dir
        self.config = Config.fromfile(config)
        if log_dir:
            self.config.work_dir = log_dir
            self.config.visualizer = dict(
                type="DetLocalVisualizer",
                vis_backends=[
                    dict(
                        type="TensorboardVisBackend",
                        save_dir=self.log_dir + "/logs",
                    )
                ],
            )

    def fit(self):
        runner = Runner.from_cfg(self.config)
        # self.test()
        runner.train()
        # os.rename(os.path.join(self.log_dir, f"best_coco_bbox_mAP_epoch_{self.config.train_cfg.max_epochs}.pth"),
        #          os.path.join(self.log_dir, "model.pth"))
        mse, accuracies = self.test()
        print("MSE: ", mse)
        print(
            "; ".join(
                f"LOC_ACC_{r}: {a}"
                for r, a in zip([25, 50, 75, 100], accuracies)
            )
        )

    def test(self) -> Tuple[float, List[float]]:
        annotations = json.load(
            open(
                self.config.test_dataloader.dataset.data_root
                + "/"
                + self.config.test_dataloader.dataset.ann_file
            )
        )
        locations, pred_locations = [], []
        losses = []
        for image in annotations["images"]:
            annotation = annotations["annotations"][image["id"]]
            image = (
                self.config.test_dataloader.dataset.data_root
                + "/"
                + self.config.test_dataloader.dataset.data_prefix.img
                + image["file_name"]
            )
            result = apis.inference_detector(self.model, image)
            location = (
                torch.Tensor(annotation["bbox"][:2])
                + torch.Tensor(annotation["bbox"][2:]) / 2
            )
            pred_location = (
                result.pred_instances.bboxes[0, :2]
                + result.pred_instances.bboxes[0, 2:]
            ) / 2
            pred_location = pred_location.cpu().detach()
            score = result.pred_instances.scores[0]
            locations.append(location)
            pred_locations.append(pred_location)
            losses.append(mse_loss(location, pred_location))
        accuracies = [
            localization_accuracy(
                torch.stack(locations), torch.stack(pred_locations), r
            )
            for r in (25, 50, 75, 100)
        ]
        return np.mean(losses), accuracies

    def inference(self, image: Image.Image) -> np.array:
        result = apis.inference_detector(self.model, np.array(image))
        pred_location = (
            result.pred_instances.bboxes[0, :2]
            + result.pred_instances.bboxes[0, 2:]
        ) / 2
        return pred_location.cpu().detach().numpy()

    def process_image(self, image: Image.Image) -> Image.Image:
        result = apis.inference_detector(self.model, np.array(image))
        location = (
            result.pred_instances.bboxes[0, :2]
            + result.pred_instances.bboxes[0, 2:]
        ) / 2
        x, y = location.cpu().detach().numpy().tolist()
        draw = ImageDraw.Draw(image)
        draw.line([(x - 60, y), (x + 60, y)], fill="blue", width=12)
        draw.line([(x, y - 60), (x, y + 60)], fill="blue", width=12)
        return image

    def load_model(self) -> None:
        self.model = apis.init_detector(
            self.config,
            os.path.join(
                "/".join(self.config_path.split("/")[:-1]), "model.pth"
            ),
        )
