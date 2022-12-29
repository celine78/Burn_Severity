import cv2
import glob
import numpy as np
import pandas as pd
from typing import List
from PIL import Image
from skimage import color, filters, io
import torch
import torch.nn.functional as fn



class Preprocessing(object):

    @staticmethod
    def load_dataset(images_path: str, masks_path: str):
        masks_dir = [file_names for file_names in glob.glob(masks_path)]
        aoi_names = [file_name[47:-5] for file_name in masks_dir]
        images_dir = [dir_name for dir_name in glob.glob(images_path) for aoi in aoi_names if aoi in dir_name]
        images_dir.sort()
        masks_dir.sort()
        return images_dir, masks_dir

    @staticmethod
    def resize(image_dir, mask_dir):
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
        return image, mask

    @staticmethod
    def get_mean_std(images):
        """
        Mean / Unit variance
        """
        images = torch.Tensor(np.array(images))
        mean = images.mean(dim=[0, 1, 2])
        std = images.std(dim=[0, 1, 2])

        return mean, std

    @staticmethod
    def get_min_max(images):
        min_value = []
        max_value = []
        for c in range(images.size()[3]):
            min_value.append(images[:, :, :, c].min())
            max_value.append(images[:, :, :, c].max())
        return min_value, max_value

    @staticmethod
    def _mask_multiclass_thresholding(mask, classes: int, index):
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

        mask_np = 1 - mask.squeeze()
        histogram = np.histogram(mask_np)[1]
        thresholds = filters.threshold_multiotsu(mask_np, int(mask_classes[index]), histogram)
        mask = np.digitize(mask_np, bins=thresholds)
        pre, nm1 = np.unique(mask, return_counts=True)
        if classes == 4 and isinstance(merge[index], str):
            cl = merge[index].split(',')
            mask[mask == int(cl[0])] = int(cl[1])
        new_classes = classification[index].split(',')
        new_classes.reverse()
        for new_value, old_value in zip(new_classes, np.unique(mask)[::-1]):
            mask[mask == old_value] = int(new_value)
        post, nm2 = np.unique(mask, return_counts=True)
        if len(pre) != len(post) and not isinstance(merge[index], str):
            print('unique pre: ', pre)
            print('unique post: ', post)
            print(nm1, nm2)
        # mask = torch.Tensor(mask)[:,:,None]
        return mask

    @staticmethod
    def _mask_binary_thresholding(mask):
        mask = torch.Tensor(mask)[:, :, None]
        mask_np = 1 - mask.squeeze().numpy()
        histogram = np.histogram(mask_np)[1]
        threshold = filters.threshold_otsu(mask_np, histogram)
        mask = (1 - mask >= threshold).int()
        mask = mask.squeeze().numpy()
        return mask

    def mask_thresholding(self, mask, classes: int, index):
        if classes == 2:
            mask = self._mask_binary_thresholding(mask)
        else:
            mask = self._mask_multiclass_thresholding(mask, classes, index)
        return mask


class DeleteLandsatBands(object):
    def __init__(self, channels: List[int]):
        self.channels = channels

    def __call__(self, image, mask):
        bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        if len(set(self.channels) - set(bands)) != 0:
            print(f'Band(s) {set(self.channels) - set(bands)} are not part of Landsat 8 & 9')
        self.channels.sort(reverse=True)
        for c in self.channels:
            image = np.delete(image, c - 1, axis=2)
        return image, mask


class Normalize(object):

    def __init__(self, mean=None, std=None, minimum=None, maximum=None, images=None):
        self.mean = mean
        self.std = std
        self.min = minimum
        self.max = maximum
        self.images = images

    def __call__(self, image, mask):
        image = torch.Tensor(np.array(image))
        if self.mean is not None and self.std is not None:
            return self._mean_std_norm(image, self.mean, self.std), mask
        elif self.min is not None and self.max is not None:
            return self._min_max_norm(image, self.min, self.max), mask
        elif self.images is not None:
            return self._l2_norm(self.images), mask
        else:
            print('Missing values for Normalization/Standardization')
            return

    @staticmethod
    def _mean_std_norm(image, mean, std):
        """
        Mean / Unit variance
        """
        for c in range(image.size()[2]):
            image[:, :, c] = (image[:, :, c] - mean[c]) / std[c]
        return image

    @staticmethod
    def _min_max_norm(image, min_value, max_value):
        for c in range(image.size()[2]):
            image[:, :, c] = (image[:, :, c] - min_value) / (max_value - min_value)
        return image

    @staticmethod
    def _l2_norm(images):
        fn.normalize(images, dim=0)
        return images
