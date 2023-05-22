import os

import mmcv
from mmdetection.mmdet.apis import (inference_detector, init_detector,
                                    show_result_pyplot, train_detector)
from mmdetection.mmdet.datasets import build_dataset
from mmdetection.mmdet.models import build_detector

from exp_runner.config import (Config, DataConfig, DatasetConfig,
                               LearningRateSchedulerConfig, LoggerConfig,
                               LossConfig, ModelConfig, OptimizerConfig,
                               PipelineConfig, TransformationsConfig,
                               TransformConfig)

SSD_CONFIG = "mmdetection/configs/ssd/ssd512_coco.py"


class Pipeline:
    def _init__(self, config: Config, log_dir: os.PathLike) -> None:
        self.config = mmcv.Config.fromfile(SSD_CONFIG)
        self.config.work_dir = log_dir
        self.load_from = "ssd/mmdetection/checkpoints/ssd512.pth"

        self.datasets = dict()
        for part in ("train", "valid", "test"):
            self.datasets[part] = build_dataset(self.config.data[part])

        self.model = build_detector(
            self.config.model,
            train_cfg=self.config.get("train_cfg"),
            valid_cfg=self.config.get("valid_cfg"),
            test_cfg=self.config.get("test_cfg"),
        )
        self.model.CLASSES = self.datasets["train"].METAINFO["classes"]

    def __configure_pipeline(self, pipeline_config: PipelineConfig) -> None:
        self.config.device = pipeline_config.device
        self.config.runner.max_epochs = pipeline_config.epochs

    def __configure_data(self, data_config: DataConfig) -> None:
        self.config.data_root = data_config.data_root
        self.config.dataset_type = "FundusDataset"
        classes = ("optic_disc",)

        self.data.samples_per_gpu = data_config.train.batch_size

        self.config.data.train.dataset.img_prefix = data_config.train.path
        self.config.data.train.dataset.classes = classes
        self.config.data.train.dataset.ann_file = os.path.join(
            data_config.train.path, data_config.train.annotations
        )

        self.config.data.valid.dataset.img_prefix = data_config.valid.path
        self.config.data.valid.dataset.classes = classes
        self.config.data.valid.dataset.ann_file = os.path.join(
            data_config.valid.path, data_config.valid.annotations
        )

        self.config.data.test.dataset.img_prefix = data_config.test.path
        self.config.data.test.dataset.classes = classes
        self.config.data.test.dataset.ann_file = os.path.join(
            data_config.test.path, data_config.test.annotations
        )

    def __configure_transforms(
        self, transforms_config: TransformationsConfig
    ) -> None:
        pass

    def __configure_optimizer(self, optimizer_config: OptimizerConfig) -> None:
        self.config.optimizer = dict(
            type=optimizer_config.type, **optimizer_config.params
        )

    def fit(self):
        train_detector(
            self.model,
            self.datasets.values(),
            self.config,
            distributed=False,
            validate=True,
        )

    def evaluate(self, image_path: os.PathLike):
        image = mmcv.imread(image_path)
        result = inference_detector(self.model, image)
        show_result_pyplot(self.model, image, result, score_thr=0.8)
