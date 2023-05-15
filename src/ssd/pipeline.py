import os

import mmcv

from exp_runner.config import (Config, DataConfig, DatasetConfig,
                               LearningRateSchedulerConfig, LoggerConfig,
                               LossConfig, ModelConfig, OptimizerConfig,
                               PipelineConfig, TransformationsConfig,
                               TransformConfig)

SSD_CONFIG = "mmdetection/configs/ssd/ssd512_coco.py"


class Pipeline:
    def _init__(self, config: Config, log_dir: os.PathLike) -> None:
        self.config = mmcv.Config.fromfile(SSD_CONFIG)

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
