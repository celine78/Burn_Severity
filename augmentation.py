import random
import math
import numpy
import numpy as np
import torch
from skimage import transform
from typing import List, Tuple

import logging.config

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RandomVflip(object):

    def __init__(self, proba: float) -> None:
        self.proba = proba

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if torch.rand(1) < self.proba:
            image = np.flipud(image)
            mask = np.flipud(mask)
            logger.debug(f'Image and mask flipped vertically')
        return image, mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + '.Proba:' + str(self.proba)


class RandomHflip(object):

    def __init__(self, proba: float) -> None:
        self.proba = proba

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if torch.rand(1) < self.proba:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            logger.debug(f'Image and mask flipped horizontally')
        return image, mask

    def __repr__(self) -> str:
        return self.__class__.__name__ + '.Proba:' + str(self.proba)


class RandomRotation(object):

    def __init__(self, proba: float, angle: int, random_angle: bool = True) -> None:
        self.proba = proba
        self.angle = angle
        self.random_angle = random_angle
        self.i = 0

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
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

    def __init__(self, proba: float, shear: int, random_shear=True) -> None:
        self.proba = proba
        self.shear = shear
        self.random_shear = random_shear

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
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

    @staticmethod
    def data_augmentation(proba_hflip: float, proba_vflip: float, proba_rotation: float, proba_shear: float, angle: int,
                          shear: int, random_angle: bool = True, random_shear: bool = True) -> \
            Tuple[RandomVflip, RandomHflip, RandomRotation, RandomShear]:
        """
        """
        return RandomVflip(proba_hflip), RandomHflip(proba_vflip), \
            RandomRotation(proba_rotation, angle, random_angle), RandomShear(proba_shear, shear, random_shear)


class Compose(object):
    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
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
