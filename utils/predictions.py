import glob
import os

import preprocessing as p
from preprocessing import Normalize
import numpy as np
import cv2
from skimage import io, color
import torch
import matplotlib.pyplot as plt


class Predictions:
    def __init__(self):
        pass

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize a satellite image
        :param image: satellite image
        :return: resized satellite image
        """
        assert type(image).__module__ == np.__name__
        image = image.astype('float64')
        image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        return image

    def __preprocess(self, image_path, level='l1'):
        prep = p.Preprocessing()
        image = io.imread(image_path)
        img_resized = self.resize_image(image)
        if level == 'l1':
            img_bands = prep.delete_landsat_bands(img_resized, [9, 10, 11, 12, 13], level='L1')
        else:
            img_bands = prep.delete_landsat_bands(img_resized, [9, 10], level='L2')
        img = torch.Tensor(np.array(img_bands))
        norm = Normalize()
        if level == 'l1':
            img_trans = norm._mean_std_norm(img, mean=torch.Tensor(
                [1166.1479, 990.1230, 860.5953, 818.1750, 2062.4404, 1656.0735,
                 1043.4313, 830.8372, 270.5279, 269.8681]),
                                            std=torch.Tensor(
                                                [445.0345, 427.1914, 453.4088, 571.3936, 1052.1285, 1009.8184,
                                                 764.5239, 492.1391, 86.5190, 86.3010]))
        else:
            img_trans = norm._mean_std_norm(img, mean=torch.Tensor([1272.2277, 1349.1849, 1535.7498, 1601.3531,
                                                                    2848.6067, 2139.2922, 1521.2327, 63552.8867]),
                                            std=torch.Tensor([2544.0225, 2525.0737, 2342.7432, 2350.0071, 2098.9248,
                                                              1357.7260, 1135.6705, 11205.6895]))
        image = img_trans.permute(2, 0, 1)
        image = image[None, :]
        return image

    def predict(self, model_path=None, image_path=None):
        image = self.__preprocess(image_path, level='l2')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        pred = model(image)
        return image, pred


if __name__ == '__main__':
    predi = Predictions()
    models = glob.glob('../models/*')
    masks = glob.glob('../../Desktop/Project/Data/WSL/vLayers/*')
    images = glob.glob('../../Desktop/Project/Data/WSL/Landsat_images_level2_2W_TI_leastRecent/*/*/response.tiff')
    masks.sort()
    images.sort()
    fig, ax = plt.subplots(12, 3, figsize=(10, 18))
    i = 0
    for model in models:
        if os.path.isdir(model): continue
        name = model[model.index('10k_') + 4: model.index('_2_')]
        for mask, image in zip(masks, images):
            image, pred = predi.predict(model_path=model, image_path=image)
            mask = io.imread(mask)
            mask = cv2.resize(mask, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            mask = color.rgba2rgb(mask)
            mask = color.rgb2gray(mask)
            img = image.squeeze().permute(1, 2, 0)[:, :, 5].detach().numpy()
            img = np.float64(img)
            img = ((img - np.min(img)) / (np.max(img) - np.min(img)))
            mask_over_image = cv2.addWeighted(img, 0.3, mask, 0.9, 0)
            ax[i][0].set_title(f'Satellite  image')
            ax[i][0].imshow(image.squeeze().permute(1, 2, 0)[:, :, 5])
            ax[i][1].set_title(f'Prediction: {name}')
            ax[i][1].imshow(pred.squeeze().permute(1, 2, 0)[:, :, 0].detach().numpy())
            ax[i][2].set_title(f'Mask over image')
            ax[i][2].imshow(mask_over_image)
            i += 1
    plt.savefig('predictions.jpg')
