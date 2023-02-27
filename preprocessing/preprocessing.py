import cv2
import glob

import numpy
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, filters, io
import torch
from typing import List, Tuple

import logging.config

logging.config.fileConfig('../logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Preprocessing(object):

    @staticmethod
    def load_dataset(images_path: str, masks_path: str) -> Tuple[List[str], List[str]]:
        masks_dir = [file_names for file_names in glob.glob(masks_path)]
        aoi_names = [file_name[47:-5] for file_name in masks_dir]
        images_dir = [dir_name for dir_name in glob.glob(images_path) for aoi in aoi_names if aoi in dir_name]
        images_dir.sort()
        masks_dir.sort()
        logger.debug(f'{len(images_dir)} images and {len(masks_dir)} masks retrieved')
        logger.info(f'Data loaded with {len(images_dir)} images')
        return images_dir, masks_dir

    @staticmethod
    def resize(image_dir: str, mask_dir: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
        mask = Image.open(mask_dir)
        image = io.imread(image_dir)
        image_height = int(image.shape[0])
        image_width = int(image.shape[1])
        mask = mask.resize((image_width, image_height))
        mask = np.array(mask)
        mask = color.rgba2rgb(mask)
        mask = color.rgb2gray(mask)
        image = image.astype('float64')
        mask = cv2.resize(mask, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        #logger.debug(f'Image shape resized: {image.shape}')
        #logger.debug(f'Mask shape resized: {mask.shape}')
        return image, mask


    @staticmethod
    def resize_image(image: numpy.ndarray) -> numpy.ndarray:
        assert type(image).__module__ == np.__name__
        image = image.astype('float64')
        image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def get_mean_std(images: List[numpy.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean / Unit variance
        """
        images = torch.Tensor(np.array(images))
        mean = images.mean(dim=[0, 1, 2])
        std = images.std(dim=[0, 1, 2])

        return mean, std

    @staticmethod
    def get_min_max(images: List[numpy.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        min_value = []
        max_value = []
        images = torch.Tensor(np.array(images))
        for c in range(images.size()[3]):
            min_value.append(images[:, :, :, c].min())
            max_value.append(images[:, :, :, c].max())
        logger.debug(f'Min: {min_value}, max: {max_value}')
        min_value = torch.Tensor(min_value)
        max_value = torch.Tensor(max_value)
        return min_value, max_value

    @staticmethod
    def _mask_multiclass_thresholding(mask: numpy.ndarray, classes: int, index: int) -> numpy.ndarray:
        logger.debug(f'Handling mask with index {index}')
        csv_data = pd.read_csv('/Users/celine/Desktop/aoi_data.csv', header=0)
        csv_data = csv_data[csv_data.notna()]
        mask_classes = csv_data['Classes']
        if classes == 4:
            merge = csv_data['Merging classes-4']
            classification = csv_data['Classification-4']
        elif classes == 5:
            merge = None
            classification = csv_data['Classification-5']
        else:
            merge = None
            classification = None
            print('Number of classes not supported')

        logger.debug(f'Merge: {merge}, classification: {classification}')

        mask_np = 1 - mask.squeeze()
        histogram = np.histogram(mask_np)[1]
        logger.debug(f'Histogram: {histogram}')
        thresholds = filters.threshold_multiotsu(mask_np, int(mask_classes[index]), histogram)
        logger.debug(f'Thresholds: {thresholds}')
        mask = np.digitize(mask_np, bins=thresholds)
        logger.debug(f'mMask: {mask}')
        pre, nm1 = np.unique(mask, return_counts=True)
        logger.debug(f'pre: {pre}, nm1: {nm1}')
        if classes == 4 and isinstance(merge[index], str):
            cl = merge[index].split(',')
            logger.debug(f'cl {cl}')
            mask[mask == int(cl[0])] = int(cl[1])
        logger.debug(f'Mask: {mask}')
        new_classes = classification[index].split(',')
        logger.debug(f'New classes: {new_classes}')
        new_classes.reverse()
        for new_value, old_value in zip(new_classes, np.unique(mask)[::-1]):
            mask[mask == old_value] = int(new_value)
        post, nm2 = np.unique(mask, return_counts=True)
        logger.debug(f'Pre: {pre}, post: {post}')
        logger.debug(f'nm1: {nm1}, nm2: {nm2}')
        if len(pre) != len(post) and not isinstance(merge[index], str):
            print('unique pre: ', pre)
            print('unique post: ', post)
            print(nm1, nm2)
        # mask = torch.Tensor(mask)[:,:,None]
        return mask

    @staticmethod
    def _mask_binary_thresholding(mask: numpy.ndarray) -> numpy.ndarray:
        mask = torch.Tensor(mask)[:, :, None]
        mask_np = 1 - mask.squeeze().numpy()
        histogram = np.histogram(mask_np)[1]
        logger.debug(f'histogram: {histogram}')
        threshold = filters.threshold_otsu(mask_np, histogram)
        logger.debug(f'threshold: {threshold}')
        mask = (1 - mask >= threshold).int()
        mask = mask.squeeze().numpy()
        return mask

    @staticmethod
    def mask_binary_thresholding_li(mask: numpy.ndarray) -> numpy.ndarray:
        mask = torch.Tensor(mask)[:, :, None]
        mask_np = 1 - mask.squeeze().numpy()
        histogram = np.histogram(mask_np)[1]
        logger.debug(f'histogram: {histogram}')
        threshold = filters.threshold_li(mask_np, initial_guess=filters.threshold_otsu(mask_np, histogram))
        logger.debug(f'threshold: {threshold}')
        mask = (1 - mask >= threshold).int()
        mask = mask.squeeze().numpy()
        return mask

    def mask_thresholding(self, mask: numpy.ndarray, classes: int, index: int) -> numpy.ndarray:
        logger.debug(f'classes {classes}, index: {index}')
        if classes == 2:
            mask = self._mask_binary_thresholding(mask)
        else:
            mask = self._mask_multiclass_thresholding(mask, classes, index)
        return mask

    @staticmethod
    def delete_landsat_bands(image: numpy.ndarray, channels: List[int]) -> numpy.ndarray:
        bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        if len(set(channels) - set(bands)) != 0:
            logger.warning(f'Band(s) {set(channels) - set(bands)} are not part of Landsat 8 & 9')
        channels.sort(reverse=True)
        for c in channels:
            image = np.delete(image, c - 1, axis=2)
        #logger.debug(f'Image shape: {image.shape}')
        return image


class Normalize(object):

    def __init__(self, mean: torch.Tensor = None, std: torch.Tensor = None, minimum: torch.Tensor = None,
                 maximum: torch.Tensor = None) -> None:
        self.mean = mean
        self.std = std
        self.min = minimum
        self.max = maximum
        logger.debug(f'mean: {mean}, std: {std}, minimum: {minimum}, maximum: {maximum}')

    def __call__(self, image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[torch.Tensor, numpy.ndarray]:
        image = torch.Tensor(np.array(image))
        if self.mean is not None and self.std is not None:
            return self._mean_std_norm(image, self.mean, self.std), mask
        elif self.min is not None and self.max is not None:
            return self._min_max_norm(image, self.min, self.max), mask
        else:
            print('Missing values for Normalization')

    def __repr__(self):
        return self.__class__.__name__ + '.Mean:' + str(self.mean) + '.Std:' + str(self.std) + '.Min:' \
            + str(self.min) + '.Max:' + str(self.max)

    @staticmethod
    def _mean_std_norm(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Mean / Unit variance
        """
        for c in range(image.size()[2]):
            image[:, :, c] = (image[:, :, c] - mean[c]) / std[c]
        return image

    @staticmethod
    def _min_max_norm(image: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        for c in range(image.size()[2]):
            image[:, :, c] = (image[:, :, c] - min_value) / (max_value - min_value)
        return image

