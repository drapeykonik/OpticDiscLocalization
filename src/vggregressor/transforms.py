"""File contains classes for different transforms of the dataset images"""
from typing import Any, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


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

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample
        image = F.crop(image, self.top, self.left, self.height, self.width)
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

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample

        w, h = image.size
        new_h, new_w = self.output_size
        image = F.resize(image, (new_h, new_w))
        location = location * [new_w / w, new_h / h]
        return image, location


class RandomFlip(object):
    """
    Class implements random horizontal flip transformation of the image.
    Provides image flipping and localization's mark coordinates modifying

    Parameters
    ----------
    p: float
        Probability of flipping
    """

    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample
        point = torch.randint(high=1, size=(1,))
        if point[0] < self.p:
            w, h = image.size
            location = (np.array([w, 0]) - location) * np.array([[1, -1]])
            image = F.hflip(image)
        return image, location


class RandomRotation(object):
    """
    Class implements random rotation of the image. Provides
    image rotating and localization's mark coordinates modifying

    Parameters
    ----------
    degrees: Union[float, Tuple[float, float]]
        Rotation angle (degrees) range. If it's tuple, angle will be random number from this tuple range.
        If it's number, angle will be random number from range [-degrees, degrees]
    """

    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        assert isinstance(degrees, (float, tuple))
        if isinstance(degrees, float):
            self.degrees = (-degrees, degrees)
        else:
            assert len(degrees) == 2
            self.degrees = degrees

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample
        w, h = image.size
        rotation = (
            torch.rand(1).item() * (self.degrees[0] - self.degrees[1])
            + self.degrees[1]
        )
        image = F.rotate(image, rotation)
        rotation = rotation / 180 * np.pi
        rotation_matrix = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        location = (
            location - np.array([[w / 2, h / 2]])
        ) @ rotation_matrix + np.array([[w / 2, h / 2]])
        return image, location


class ColorChannel(object):
    """
    Class implements color channel choosing transformation

    Parameters
    ----------
    channel: int
        Number of channel to choose. Must be equal 0, 1, 2
    """

    def __init__(self, channel: int):
        assert isinstance(channel, int)
        assert 0 <= channel < 3
        self.channel = channel

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample
        channels = image.split()
        image = Image.merge("L", (channels[self.channel],))
        return image, location


class GrayScale(object):
    """
    Class implements transformation of the image to grayscale
    """

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[PIL.Image.Image, numpy.array]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample
        image = image.convert("L")
        return image, location


class ToTensor(object):
    """
    Class implements transformation of the PIL image
    to tensors
    """

    def __call__(self, sample: Tuple[Image.Image, np.array]):
        """
        Transformation logic

        Parameters
        ----------
        sample: Tuple[Any, Any]
            Sample to transform (image and localization's mark coordinates)
        """
        image, location = sample
        image = F.to_tensor(image)
        location = torch.from_numpy(location)
        return image, location
