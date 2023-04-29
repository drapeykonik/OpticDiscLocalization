"""File for experiment config defining"""
from typing import Dict, List

from pydantic import BaseModel


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
