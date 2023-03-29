import os
import glob

import numpy
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from utils.augmentation import Compose
from utils import Preprocessing

import logging.config

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    def __init__(self, images: List[numpy.ndarray], masks: numpy.ndarray, class_num: int = None,
                 transform: Compose = None, save_masks: bool = False) -> None:
        """
        A dataset class which preprocesses and augments the data for training purposes using
        :param images: satellite images
        :param masks: masks
        :param class_num: number of classes
        :param transform: transformations to perform
        :param save_masks: whether to save the masks
        """
        self.images = images
        self.masks = masks
        self.class_num = class_num
        self.transform = transform
        self.save_masks = save_masks

    def __len__(self) -> int:
        """
        Computes the length of the dataset
        :return: length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocesses and augments an image and its mask
        :param index: index of the image and mask
        :return: the image and its mask
        """
        image = self.images[index]
        mask = self.masks[index]
        logger.debug(f'image, mask length: {len(image), len(mask)}, Index: {index}')
        prep = Preprocessing()
        mask = prep.mask_thresholding(mask, self.class_num, index)
        logger.debug(f'mask shape: {mask.shape}')
        if self.save_masks:
            self.save_mask(mask, self.class_num, index)
        if self.transform is not None:
            #logger.debug(f'Image min/max before normalization {image.min(), image.max()}')
            image, mask = self.transform(image, mask)
            #logger.debug(f'Image min/max after normalization: {image.min(), image.max()}')
        image = image.permute(2, 0, 1)
        logger.debug(f'Image: {image.size()}')
        logger.debug(f'Mask: {mask.size()}')

        return image, mask

    @classmethod
    def save_mask(cls, mask: numpy.ndarray, class_num: int, index: int) -> None:
        """
        Saves a mask to a local directory
        :param mask: mask to save
        :param class_num: number of classes
        :param index: index of the mask
        """
        mask_files = [file_names for file_names in glob.glob('Thesis/Data/Maps/vLayers/*.tiff')]
        mask_files.sort()
        mask = mask.cpu().detach().numpy()
        os.makedirs(f'/Users/celine/Desktop/masks_{class_num}_classes', exist_ok=True)
        mask.imsave(f'/Users/celine/Desktop/masks_{class_num}_classes/' + mask_files[index][25:-5]
                    + '.tiff', mask[:, :, 0], format='tiff', cmap='gray')
