"""File for experiment config defining"""
import json
from typing import Dict, List, Type, TypeVar

from pydantic import BaseModel
from yaml import safe_load


class DatasetConfig(BaseModel):
    """
    Class for defining dataset config

    Parameters
    ----------
    data_root: str
        Path to the data folder
    train: str
        Path to the training data
    valid: str
        Path to the validating data
    test: str
        Path to the testing data
    train_ann: str
        Path to the annotation file for the training data
    valid_ann: str
        Path to the annotation file for the validating data
    test_ann: str
        Path to the annotation file for the testing data
    train_batch: int
        Size of the mini-batch for the training dataset (dataloader)
    valid_batch: int
        Size of the mini-batch for the validating dataset (dataloader)
    test_batch: int
        Size of the mini-batch for the testing dataset (dataloader)
    """

    data_root: str = "data/processed/localization"
    train: str = "data/processed/localization/train"
    valid: str = "data/processed/localization/valid"
    test: str = "data/processed/localization/test"
    train_ann: str = "data/processed/localization/train/location.json"
    valid_ann: str = "data/processed/localization/valid/location.json"
    test_ann: str = "data/processed/localization/test/location.json"
    train_batch: int = 8
    valid_batch: int = 1
    test_batch: int = 1


class TransformConfig(BaseModel):
    """
    Class for defining transform config

    Parameters
    ----------
    type: str
        Type name of the transform
    params: Dict
        Params for the transformation. Exactly parameters
        depends on the type of the transformation

    """

    transform: str
    params: Dict


class TransformationsConfig(BaseModel):
    """
    Class for defining tranformations config for
    each dataset part

    Parameters
    ----------
    train: List[TransformConfig]
        Transform pipeline for the train dataset
    valid: List[TransformConfig]
        Transform pipeline for the valid dataset
    test: List[TransformConfig]
        Transform pipeline for the test dataset
    """

    train: List[TransformConfig]
    valid: List[TransformConfig]
    test: List[TransformConfig]


class ModelConfig(BaseModel):
    """
    Class for defining model config

    Parameters
    ----------
    name: str
        Model name
    """

    name: str


class LossConfig(BaseModel):
    """
    Class for defining loss function config

    Parameters
    ----------
    type: str
        Name of the loss function
    params: Dict
        Dict of params for specified loss function
    """

    type: str
    params: Dict


class OptimizerConfig(BaseModel):
    """
    Class for defining optimizer config for
    model training

    Parameters
    ----------
    type: str
        Name of the optimizer
    params: Dict
        Dictionary of params for optimizer
    """

    type: str
    params: Dict


class LearningRateSchedulerConfig(BaseModel):
    """
    Class for defining learning rate scheduler

    Parameters
    ----------
    type: str
        Name of the learning rate scheduler
    params: Dict
        Dictionary of params for learning rate scheduler
    """

    type: str
    params: Dict


class LoggerConfig(BaseModel):
    """
    Class for defining experiments logger

    Parameters
    ----------
    path: str
        Path to logs folder
    """

    path: str


T = TypeVar("T", bound="Config")


class Config(BaseModel):
    """
    Class for whole config for experiment

    Parameters
    ----------
    data: DatasetConfig
        Config part for dataset
    transforms: TransformationsConfig
        Config part for transformations
    model: ModelConfig
        Config part for model
    loss: LossConfig
        Config part for loss function
    optimizer: OptimizerConfig
        Config part for optimizer
    lr_scheduler: LearningRateSchedulerConfig
        Config part for learning rate scheduler
    logger: LoggerConfig
        Config part for logger
    """

    data: DatasetConfig
    transforms: TransformationsConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    lr_scheduler: LearningRateSchedulerConfig
    logger: LoggerConfig

    @classmethod
    def parse(cls: Type[T], path: str) -> T:
        """
        Class method for parsing yaml config specified by path

        Parameters
        ----------
        path: str
            Path to yaml config file
        """
        with open(path, "r") as yaml_cfg:
            cfg = cls.parse_raw(json.dumps(safe_load(yaml_cfg)))
        return cfg
