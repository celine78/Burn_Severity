import random
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import transform

import logging.config

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class DataAugmentation(object):

    @staticmethod
    def data_augmentation(proba_hflip, proba_vflip, proba_rotation, proba_shear, angle, shear, random_angle=True,
                          random_shear=True):
        """
        """
        return RandomVflip(proba_hflip), RandomHflip(proba_vflip), \
            RandomRotation(proba_rotation, angle, random_angle), RandomShear(proba_shear, shear, random_shear)


class RandomVflip(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, image, mask):
        if torch.rand(1) < self.proba:
            image = np.flipud(image)
            mask = np.flipud(mask)
            logger.debug(f'Image and mask flipped vertically')
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '.Proba:' + str(self.proba)


class RandomHflip(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, image, mask):
        if torch.rand(1) < self.proba:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            logger.debug(f'Image and mask flipped horizontally')
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '.Proba:' + str(self.proba)


class RandomRotation(object):

    def __init__(self, proba, angle, random_angle=True):
        self.proba = proba
        self.angle = angle
        self.random_angle = random_angle
        self.i = 0

    def __call__(self, image, mask):
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
            logger.debug(f'Angle: {self.angle_applied}')
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '.Angle:' + str(self.angle_applied) + '.Proba:' + str(self.proba)


class RandomShear(object):

    def __init__(self, proba, shear, random_shear=True):
        self.proba = proba
        self.shear = shear
        self.random_shear = random_shear

    def __call__(self, image, mask):
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
            logger.debug(f'Shear: {shear}')
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '.Shear:' + str(self.shear_applied) + '.Proba:' + str(self.proba)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        img = image.copy()
        mask_pre = mask.copy()
        for t in self.transforms:
            image, mask = t(image, mask)
        logger.debug(f'Image transformed: {np.array_equal(img,image) == False}')
        logger.debug(f'Mask transformed: {np.array_equal(mask_pre,mask) == False}')
        return image, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f" {t}"
        format_string += "\n)"
        return format_string
