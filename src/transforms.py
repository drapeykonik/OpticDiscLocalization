"""File contains classes for different transforms of the dataset images"""
from typing import Any, Tuple

import numpy as np
from torchvision.transforms.functional import crop


class Crop(object):
    """
    Class for cropping sample from dataset. This transformation provides
    image cropping and localization's mark modifying

    Parameters
    ----------
    top: int
        Y-coordinate of top left corner of the cropping rectangle
    left: int
        X-coordinate of the top left corner of the cropping rectangle
    height: int
        Height of the cropping rectangle
    width: int
        Width of the cropping rectangle
    """

    def __init__(self, top: int, left: int, height: int, width: int):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, sample: Tuple[Any, Any]):
        image, location = sample
        image = crop(image, self.top, self.left, self.height, self.width)
        location = location - np.array([[self.left, self.top]])
        return image, location
