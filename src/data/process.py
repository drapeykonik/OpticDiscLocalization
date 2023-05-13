"""File for dataset processing"""
import os
from shutil import copy

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATASET_DIRECTORY = os.path.join("data", "raw", "localization")
DATASET_LABELS_FILENAME = "location.csv"
PROCESSED_DATASET_DIRECTORY = os.path.join("data", "processed", "localization")
NEW_LABELS_COLUMNS = ["Image", "X", "Y"]


def clean_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Function for cleaning labels DataFrame
    (NaNs dropping, columns renaming)

    Parameters
    ----------
    labels: pandas.DataFrame
        pandas.DataFrame with images names and their location labels

    Returns
    -------
    labels: pandas.DataFrame
        Cleaned labels
    """

    labels = labels.dropna(axis=0, subset=labels.columns[:3]).dropna(axis=1)
    labels.columns = NEW_LABELS_COLUMNS
    labels[NEW_LABELS_COLUMNS[0]] += ".jpg"
    return labels


def copy_images(
    images_names: pd.Series, src: os.PathLike, dest: os.PathLike
) -> None:
    """
    Function to copy images for source to destination directory

    Parameters
    ----------
    images_name: pd.Series
        pandas.Series with images names to copy
    src: os.PathLike
        Source directory path
    dest: os.PathLike
        Destination directory path

    Returns
    -------
    None
    """

    images_names.apply(
        lambda image: copy(os.path.join(src, image), os.path.join(dest, image))
    )


def build_data_dir(labels: pd.DataFrame, src_dir: str, dest_dir: str) -> None:
    """
    Function to build a directory for a part of a processed dataset

    Parameters
    ----------
    labels: pandas.DataFrame
        pandas DataFrame with labels of the part of the processed dataset
    src_dir: str
        Source directory name
    dest_dir: str
        Destination directory name

    Returns
    -------
    None
    """

    os.makedirs(
        os.path.join(PROCESSED_DATASET_DIRECTORY, dest_dir), exist_ok=True
    )
    labels.to_csv(
        os.path.join(
            PROCESSED_DATASET_DIRECTORY, dest_dir, DATASET_LABELS_FILENAME
        )
    )
    copy_images(
        labels["Image"],
        os.path.join(RAW_DATASET_DIRECTORY, src_dir),
        os.path.join(PROCESSED_DATASET_DIRECTORY, dest_dir),
    )


def process_data(valid_size: float = 0.2) -> None:
    """
    Function to process raw dataset (splitting train set on
    training set and validation set)

    Parameters
    ----------
    valid_size: float
        Fraction of data for splitting training data
        on training set and validation data

    Returns
    -------
    None
    """
    if not os.path.exists(PROCESSED_DATASET_DIRECTORY):
        os.makedirs(PROCESSED_DATASET_DIRECTORY, exist_ok=True)
        train_labels = pd.read_csv(
            os.path.join(
                RAW_DATASET_DIRECTORY, "train", DATASET_LABELS_FILENAME
            ),
            sep=",",
        )
        train_labels = clean_labels(train_labels)
        train_labels, valid_labels = train_test_split(
            train_labels, test_size=valid_size
        )
        train_labels.reset_index(inplace=True, drop=True)
        valid_labels.reset_index(inplace=True, drop=True)
        build_data_dir(train_labels, "train", "train")
        build_data_dir(valid_labels, "train", "valid")
        test_labels = pd.read_csv(
            os.path.join(
                RAW_DATASET_DIRECTORY, "test", DATASET_LABELS_FILENAME
            ),
            sep=",",
        )
        test_labels = clean_labels(test_labels)
        build_data_dir(test_labels, "test", "test")


if __name__ == "__main__":
    process_data(0.2)
