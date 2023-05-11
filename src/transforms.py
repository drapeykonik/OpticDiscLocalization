"""File contains classes for different transforms of the dataset images"""
from typing import Any, Tuple

import numpy as np
from torchvision.transforms.functional import crop, resize


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


class Resize(object):
    """
    Class implement transformation to resize sample from the dataset.
    Provides image resizing and localization's mark coordinates modifying

    Parameters
    ----------
    output_size: Union[int, tuple]
        Output size of the image
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            self.output_size = (output_size, output_size)

    def __call__(self, sample):
        image, location = sample

        w, h = image.size
        new_h, new_w = self.output_size
        image = resize(image, (new_h, new_w))
        location = location * [new_w / w, new_h / h]
        return image, location
