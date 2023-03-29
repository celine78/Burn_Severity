import random
import math
import numpy
import numpy as np
import torch
from skimage import transform
from typing import List, Tuple

import logging.config

logger = logging.getLogger(__name__)


class RandomVflip(object):
    """
    Random vertical flip class for an image and its mask
    """

    def __init__(self, proba: float) -> None:
        """
        Constructor of the vertical flip class
        :param proba: probability of the flip
        """
        self.proba = proba

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Vertical flip of an image and its mask
        :param image: image to flip
        :param mask: mask to flip
        :return: vertically flipped image
        """
        if torch.rand(1) < self.proba:
            image = np.flipud(image)
            mask = np.flipud(mask)
            logger.debug(f'Image and mask flipped vertically')
        return image, mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + '.Proba:' + str(self.proba)


class RandomHflip(object):
    """
    Random horizontal flip class for an image and its mask
    """
    def __init__(self, proba: float) -> None:
        """
       Constructor of the horizontal flip class
       :param proba: probability of the flip
       """
        self.proba = proba

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Horizontal flip of an image and its mask
        :param image: image to flip
        :param mask: mask to flip
        :return: horizontally flipped image
        """
        if torch.rand(1) < self.proba:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            logger.debug(f'Image and mask flipped horizontally')
        return image, mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + '.Proba:' + str(self.proba)


class RandomRotation(object):
    """
    Random rotation class for an image and its mask
    """

    def __init__(self, proba: float, angle: int, random_angle: bool = True) -> None:
        """
        Initializes the probability of rotation, the maximum angle of rotation and whether to use a random rotation
        :param proba: probability of rotation
        :param angle: maximum angle of rotation
        :param random_angle: whether to use a random rotation
        """
        self.proba = proba
        self.angle = angle
        self.random_angle = random_angle
        self.i = 0

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Rotates an image and its mask
        :param image: image to rotate
        :param mask: mask to rotate
        :return:
        """
        self.angle_applied = 0
        if torch.rand(1) < self.proba:
            if self.random_angle:
                angle = random.randint(-self.angle, self.angle)
                if angle == 0.0:
                    angle += 0.1
            else:
                angle = self.angle
            image = transform.rotate(image, angle, mode='reflect', preserve_range=True)
            mask = transform.rotate(mask, angle, mode='reflect', preserve_range=True)
            self.angle_applied = angle
            logger.debug(f'Transformed with angle: {self.angle_applied}')
        return image, mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + '.Angle:' + str(self.angle_applied) + '.Proba:' + str(self.proba)


class RandomShear(object):
    """
    Random warping class for an image and its mask given a maximum shear radius or using a random radius
    """

    def __init__(self, proba: float, shear: int, random_shear=True) -> None:
        """
        Initializes the probability of warping, the maximum shear radius and whether to use a random radius
        :param proba: probability of warping
        :param shear: shear angle
        :param random_shear: whether to use a random radius
        """
        self.proba = proba
        self.shear = shear
        self.random_shear = random_shear

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Warps an image and its mask
        :param image: image to warp
        :param mask: mask to warp
        :return: warped image and mask
        """
        self.shear_applied = 0
        if torch.rand(1) < self.proba:
            if self.random_shear:
                shear = random.randint(-self.shear, self.shear)
                if shear == 0.0:
                    shear += 0.1
            else:
                shear = self.shear
            shear = math.radians(shear)
            trans = transform.AffineTransform(shear=shear)
            image = transform.warp(image, trans, mode='reflect', preserve_range=True)
            mask = transform.warp(mask, trans, mode='reflect', preserve_range=True)
            self.shear_applied = shear
            logger.debug(f'Transformed with shear: {shear}')
        return image, mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + '.Shear:' + str(self.shear_applied) + '.Proba:' + str(self.proba)


class DataAugmentation(object):
    """
    Data augmentation class
    """

    @staticmethod
    def data_augmentation(proba_hflip: float, proba_vflip: float, proba_rotation: float, proba_shear: float, angle: int,
                          shear: int, random_angle: bool = True, random_shear: bool = True) -> \
            Tuple[RandomVflip, RandomHflip, RandomRotation, RandomShear]:
        """
        Augments the data with respects to the given parameters
        :param proba_hflip: probability of horizontal flip
        :param proba_vflip: probability of vertical flip
        :param proba_rotation: probability of rotation
        :param proba_shear: probability of warping
        :param angle: maximum rotation angle
        :param shear: maximum shear radius
        :param random_angle: whether to use a random rotation angle
        :param random_shear: whether to use a random shear radius
        :return: RandomVflip, RandomHflip, RandomRotation and RandomShear classes instance
        """
        return RandomVflip(proba_hflip), RandomHflip(proba_vflip), \
            RandomRotation(proba_rotation, angle, random_angle), RandomShear(proba_shear, shear, random_shear)


class Compose(object):
    """
    Compose class for the transformation of the data
    """
    def __init__(self, transforms: List) -> None:
        """
        Initialization with a list of transformations
        :param transforms: transformations
        """
        self.transforms = transforms

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform the given image and mask
        :param image: image to transform
        :param mask: mask to transform
        :return: image and mask transformed
        """
        for t in self.transforms:
            image, mask = t(image, mask)
        mask = torch.Tensor(mask.copy())[None, :, :]
        return image, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f" {t}"
        format_string += "\n)"
        return format_string
