"""File for experiment config defining"""
import json
from typing import Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel
from yaml import safe_load


class PipelineConfig(BaseModel):
    """
    Class for defining pipeline config

    Parameters
    ----------
    device: str
        Device name
    epochs: int
        Epochs number for training
    """

    device: str
    epochs: int


class DatasetConfig(BaseModel):
    """
    Class for defining dataset config

    Parameters
    ----------
    path: str
        Path to the data
    annotations: str
        Name of the annotation file
    batch_size:
        Number of samples in the mini-batch
    """

    path: str
    annotations: str
    batch_size: int


class DataConfig(BaseModel):
    """
    Class for defining data config

    Parameters
    ----------
    train: DatasetConfig
        Training dataset config
    valid: DatasetConfig
        Validation dataset config
    test: DatasetConfig
        Testing dataset config
    """

    train: DatasetConfig
    valid: DatasetConfig
    test: DatasetConfig


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
    params: Optional[Dict]


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
    params: Optional[Dict]


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
    params: Optional[Dict]


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
    params: Optional[Dict]


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
    params: Optional[Dict]


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
    data: DataConfig
        Config part for data
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

    pipeline: PipelineConfig
    data: DataConfig
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

        Returns
        -------
        config: Config
            Parsed config
        """
        with open(path, "r") as yaml_cfg:
            cfg = cls.parse_raw(json.dumps(safe_load(yaml_cfg)))
        return cfg
