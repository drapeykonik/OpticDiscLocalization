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
        return self.ann_frame.shape[0]

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.array]:
        image_path = os.path.join(
            self.data_root, self.ann_frame.iloc[index, 1]
        )
        image = Image.open(image_path)
        location = self.ann_frame.iloc[index, 2:]
        location = np.array([location])
        location = location.astype("float").reshape(-1, 2)
        sample = (image, location)

        if self.transform:
            sample = self.transform(sample)

        return sample
