"""File for custom dataset implementing"""
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FundusDataset(Dataset):
    """
    Class for custom dataset

    Parameters
    ----------
    data_root: os.PathLike
        Path to the root directory of the data
    ann_file: str
        Name of the annotation file
    transform: Optional[Callable]
        Transforms for dataset sample
    """

    def __init__(
        self,
        data_root: os.PathLike,
        ann_file: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_root = data_root
        self.ann_file = ann_file
        self.transform = transform
        self.ann_frame = pd.read_csv(
            os.path.join(data_root, ann_file), sep=","
        )

    def __len__(self) -> int:
        """
        Getting length of the dataset (amount of samples)

        Returns
        -------
        length: int
            Length of the dataset
        """
        return self.ann_frame.shape[0]

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.array]:
        """
        Getting dataset sample

        Parameters
        ----------
        index: int
            Index of the sample

        Returns
        -------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample from dataset (image and localization mark coordinates)
        """
        sample = self.get(index)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get(self, index: int) -> Tuple[Image.Image, np.array]:
        """
        Getting dataset samples without transforms

        Parameters
        ----------
        index: int
            Index of the sample

        Returns
        -------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample from dataset (image and localization mark coordinates)
            without transforms
        """
        image_path = os.path.join(
            self.data_root, self.ann_frame.iloc[index, 1]
        )
        image = Image.open(image_path)
        location = self.ann_frame.iloc[index, 2:]
        location = np.array(location).astype("float").reshape(-1)
        return image, location
