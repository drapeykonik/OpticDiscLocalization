"""File for experiment config defining"""
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
