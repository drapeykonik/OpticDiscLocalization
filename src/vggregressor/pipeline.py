import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tfs
from tqdm import tqdm

from exp_runner.config import (Config, DataConfig, DatasetConfig,
                               LearningRateSchedulerConfig, LoggerConfig,
                               LossConfig, ModelConfig, OptimizerConfig,
                               PipelineConfig, TransformationsConfig,
                               TransformConfig)
from vggregressor.dataset import FundusDataset


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
    log_dir: os.PathLike
        Directory for logs
    """

    def __init__(self, config: Config, log_dir: os.PathLike) -> None:
        self.config = config
        self.logger = SummaryWriter(os.path.join("experiments", log_dir))

        self.device = config.pipeline.device
        self.epochs = config.pipeline.epochs
        self.model = Pipeline.__create_model(config.model).to(self.device)
        self.data_loaders = Pipeline.__create_data_loaders(
            config.data, config.transforms
        )
        self.criterion = Pipeline.__create_loss(config.loss)
        self.optimizer = Pipeline.__create_optimizer(
            config.optimizer, self.model
        )
        self.scheduler = Pipeline.__create_scheduler(
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

        Returns
        -------
        model: nn.Module
            Created model
        """
        exec("from vggregressor.model import " + model_config.name)
        return (
            eval(model_config.name + "(**model_config.params)")
            if model_config.params is not None
            else eval(model_config.name + "()")
        )

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

        Returns
        -------
        transform: Callable
            Created transform pipeline
        """
        transforms = []
        for t in transform_configs:
            exec("from vggregressor.transforms import " + t.transform)
            transforms.append(
                (
                    eval(t.transform + "(**t.params)")
                    if t.params is not None
                    else eval(t.transform + "()")
                )
            )
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

        Returns
        -------
        dataset: FundusDataset
            Created dataset
        """
        return FundusDataset(
            data_root=dataset_config.path,
            ann_file=dataset_config.annotations,
            transform=Pipeline.__create_transform(transform_configs),
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

        Returns
        -------
        data_loaders: Dict[str, torch.utils.data.DataLoader]
            Dictionary of the data loaders for every dataset
        """
        data_loaders = dict()
        for dataset_name, dataset_config in data_config.dict().items():
            data_loaders[dataset_name] = DataLoader(
                Pipeline.__create_dataset(
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

        Returns
        -------
        loss: Callable
            Created loss function
        """
        exec("from torch.nn import " + loss_config.type)
        return (
            eval(loss_config.type + "(**loss_config.params)")
            if loss_config.params is not None
            else eval(loss_config.type + "()")
        )

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

        Returns
        -------
        optimizer: torch.optim.Optimizer
            Created optimizer
        """
        exec("from torch.optim import " + optimizer_config.type)
        return (
            eval(
                optimizer_config.type
                + "(model.parameters(), **optimizer_config.params)"
            )
            if optimizer_config.params is not None
            else eval(optimizer_config.type + "()")
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

        Returns
        -------
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Created learning rate scheduler
        """
        exec("from torch.optim.lr_scheduler import " + scheduler_config.type)
        return (
            eval(
                scheduler_config.type
                + "(optimizer, **scheduler_config.params)"
            )
            if scheduler_config.params is not None
            else eval(scheduler_config.type + "()")
        )

    def __train_epoch(self, epoch: int) -> List[float]:
        """
        Method to perform one epoch of training

        Returns
        -------
        losses: List[float]
            Losses after training epoch
        """
        train_losses_epoch = []

        self.model.train()
        for i, (images, locations) in enumerate(self.data_loaders["train"]):
            self.optimizer.zero_grad()
            pred_locations = self.model(images.to(self.device))
            loss = self.criterion(locations.to(self.device), pred_locations)
            train_losses_epoch.append(loss.item())
            self.logger.add_scalar(
                "Train loss",
                loss.item(),
                epoch * len(self.data_loaders["train"]) + i,
            )
            loss.backward()
            self.optimizer.step()
        return train_losses_epoch

    def __valid_epoch(self, epoch: int) -> List[float]:
        """
        Method to perform one epoch of validation

        Returns
        -------
        losses: List[float]
            Losses after validation epoch
        """
        valid_losses_epoch = []

        self.model.eval()
        with torch.no_grad():
            for i, (images, locations) in enumerate(
                self.data_loaders["valid"]
            ):
                pred_locations = self.model(images.to(self.device))
                loss = self.criterion(
                    locations.to(self.device), pred_locations
                )
                valid_losses_epoch.append(loss.item())
                self.logger.add_scalar(
                    "Valid loss",
                    loss.item(),
                    epoch * len(self.data_loaders["valid"] + i),
                )

        return valid_losses_epoch

    def fit(self) -> Tuple[List[float], List[float]]:
        """
        Method to perform full cycle of the training/validation of the model

        Returns
        -------
        losses: Tuple[List[float], List[float]]
            Losses after training/validation cycle from each epoch
        """
        train_losses, valid_losses = [], []
        torch.cuda.empty_cache()
        pbar = tqdm(range(self.epochs))
        pbar.set_description("Epoch 1")
        for epoch in pbar:
            if epoch != 0:
                pbar.set_description(
                    f"""Epoch {epoch + 1}. Train loss: {round(train_losses[-1], 2)}. Valid loss: {round(valid_losses[-1], 2)}"""
                )

            train_losses_epoch = self.__train_epoch()
            valid_losses_epoch = self.__valid_epoch()

            train_losses.append(np.mean(train_losses_epoch))
            valid_losses.append(np.mean(valid_losses_epoch))
            self.scheduler.step()

        return train_losses, valid_losses

    def test(self) -> None:
        """
        Method to perform testing the model
        using test dataset

        Returns
        -------
        losses: List[float]
            Losses after testing
        """
        losses = []

        self.model.eval()
        with torch.no_grad():
            for images, locations in self.data_loaders["test"]:
                pred_locations = self.model(images.to(self.device))
                loss = self.criterion(
                    locations.to(self.device), pred_locations
                )
                losses.append(loss.item())

        return losses

    def evaluate(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a model prediction for specified image

        Perameters
        ----------
        image: PIL.Image.Image
            Image for localization mark coodinates predicting

        Returns
        -------
        sample: Tuple[torch.Tensor, torch.Tensor]
            Sample (transformed image and predicted localization mark coordinates)
        """
        tfs_image, _ = self.data_loaders["test"].dataset.transform(
            (image, np.array([0, 0]))
        )
        location = self.model(image.view((-1, *image.shape)).to(self.device))
        return tfs_image, location

    def inverse_transform(
        self, target: str, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Image.Image, np.array]:
        """
        Method to inverse transformation for sample
        of the specified transfromations part (target)

        Parameters
        ----------
        targer: str
            Name of the transformation part
        sample: Tuple[torch.Tensor, torch.Tensor]
            Sample to transform

        Returns
        -------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Inverse transformed sample
        """
        for tfs in reversed(
            self.data_loaders[target].dataset.transform.transforms
        ):
            sample = tfs.inverse(sample)
        return sample

    def save_model(self, path: os.PathLike) -> None:
        """
        Method to save model by path

        Parameters
        ----------
        path: os.PathLike
            Path to save the model

        Returns
        -------
        None
        """
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
