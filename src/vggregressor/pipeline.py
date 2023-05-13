from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from tqdm import tqdm

from src.config import (Config, DataConfig, DatasetConfig,
                        LearningRateSchedulerConfig, LoggerConfig, LossConfig,
                        ModelConfig, OptimizerConfig, PipelineConfig,
                        TransformationsConfig, TransformConfig)
from src.vggregressor.dataset import FundusDataset


class Pipeline:
    """
    Class for the full pipeline.
    It's built using config file. This class creates all of
    the part of pipeline (model, datasets, transforms, data loaders,
    loss function, optimizer, learning rate scheduler, logger)
    Provides methods for training model using trainining/validation
    cycle, and model evaluation

    Parameters
    ----------
    config: Config
        Config for full pipeline
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        self.device = config.pipeline.device
        self.epochs = config.pipeline.epochs
        self.model = Pipeline._create_model(config.model).to(self.device)
        self.data_loaders = Pipeline._create_data_loaders(
            config.data, config.transforms
        )
        self.loss = Pipeline._create_loss(config.loss)
        self.optimizer = Pipeline._create_optimizer(
            config.optimizer, self.model
        )
        self.scheduler = Pipeline._create_scheduler(
            config.lr_scheduler, self.optimizer
        )
        self.logger = None

    @staticmethod
    def __create_model(model_config: ModelConfig) -> nn.Module:
        """
        Private method for model creating using model config

        Parameters
        ----------
        model_config: ModelConfig
            Model config part
        """
        exec("from src.vggregressor.model import " + model_config.name)
        return eval(model_config.name + "(**model_config.params)")

    @staticmethod
    def __create_transform(
        transform_configs: List[TransformConfig],
    ) -> Callable:
        """
        Private method for transform creating using list of
        the transform configs

        Parameters
        ----------
        transform_configs: List[TransformConfig]
            List of the transform configs
        """
        transforms = []
        for t in transform_configs:
            exec("from src.vggregressor.transforms import " + t.transform)
            transforms.append(eval(t.transform + "(**t.params)"))
        return tfs.Compose(transforms)

    @staticmethod
    def __create_dataset(
        dataset_config: DatasetConfig, transform_configs: List[TransformConfig]
    ) -> FundusDataset:
        """
        Private method for dataset creating using dataset config

        Parameters
        ----------
        dataset_config: DatasetConfig
            Dataset config
        transform_configs: List[TransformConfig]
            List of the transform configs
        """
        return FundusDataset(
            data_root=dataset_config.path,
            ann_file=dataset_config.annotations,
            transform=Pipeline._create_transform(transform_configs),
        )

    @staticmethod
    def __create_data_loaders(
        data_config: DataConfig, transforms_config: TransformationsConfig
    ) -> Dict[str, DataLoader]:
        """
        Private method for data loaders creating using data config
        and transforms config

        Parameters
        ----------
        data_config: DataConfig
            Data config
        transforms_config: TransformationsConfig
            Transformations config
        """
        data_loaders = dict()
        for dataset_name, dataset_config in data_config.dict().items():
            data_loaders[dataset_name] = DataLoader(
                Pipeline._create_dataset(
                    DatasetConfig.parse_obj(dataset_config),
                    getattr(transforms_config, dataset_name),
                ),
                batch_size=getattr(data_config, dataset_name).batch_size,
                shuffle=True if dataset_name == "train" else False,
            )
        return data_loaders

    @staticmethod
    def __create_loss(loss_config: LossConfig) -> Callable:
        """
        Private method for loss creating using loss config

        Parameters
        ----------
        loss_config: LossConfig
            Config for loss function
        """
        exec("from torch.nn import " + loss_config.type)
        return eval(loss_config.type + "(**loss_config.params)")

    @staticmethod
    def __create_optimizer(
        optimizer_config: OptimizerConfig, model: nn.Module
    ) -> torch.optim.Optimizer:
        """
        Private method for optimizer creating using optimizer
        config and models parameters

        Parameters
        ----------
        optimizer_config: OptimizerConfig
            Optimzier config
        model: nn.Module
            Model to optimize
        """
        exec("from torch.optim import " + optimizer_config.type)
        return eval(
            optimizer_config.type
            + "(model.parameters(), **optimizer_config.params)"
        )

    @staticmethod
    def __create_scheduler(
        scheduler_config: LearningRateSchedulerConfig,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Private method for learning rate scheduler creating
        using scheduler config and optimizer

        Parameters
        ----------
        scheduler_config: LearningRateSchedulerConfig
            Learning rate scheduler config
        optimizer: torch.optim.Optimizer
            Optimizer to schedule learning rate
        """
        exec("from torch.optim.lr_scheduler import " + scheduler_config.type)
        return eval(
            scheduler_config.type + "(optimizer, **scheduler_config.params)"
        )

    def __train_epoch(self) -> List[float]:
        """
        Method to perform one epoch of training
        """
        train_losses_epoch = []

        self.model.train()
        for (images, locations) in tqdm(self.data_loaders["train"]):
            self.optimizer.zero_grad()
            pred_locations = self.model(images.to(self.device).float())
            loss = self.criterion(
                locations.to(self.device).float(), pred_locations
            )
            train_losses_epoch.append(loss.item())
            loss.backward()
            self.optimizer.step()
        return train_losses_epoch

    def __valid_epoch(self) -> List[float]:
        """
        Method to perform one epoch of validation
        """
        valid_losses_epoch = []

        self.model.eval()
        with torch.no_grad():
            for images, locations in self.data_loaders["valid"]:
                pred_locations = self.model(images.to(self.device).float())
                loss = self.criterion(
                    locations.to(self.device).float(), pred_locations
                )
                valid_losses_epoch.append(loss.item())

        return valid_losses_epoch

    def fit(self) -> Tuple[List[float], List[float]]:
        """
        Method to perform full cycle of the training/validation of the model
        """
        train_losses, valid_losses = [], []
        torch.cuda.empty_cache()
        pbar = tqdm(range(self.epochs))
        pbar.set_description("Epoch 1")
        for epoch in pbar:
            if epoch != 0:
                pbar.set_description(
                    f"""Epoch {epoch + 1}.
                                         Train loss: {round(train_losses[-1], 4)}.
                                         Valid loss: {round(valid_losses[-1], 4)}"""
                )

            train_losses_epoch = self.train_epoch()
            valid_losses_epoch = self.valid_epoch()

            train_losses.append(np.mean(train_losses_epoch))
            valid_losses.append(np.mean(valid_losses_epoch))

        return train_losses, valid_losses
