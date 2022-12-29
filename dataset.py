import os
import glob
import torch
from torch.utils.data import Dataset

from preprocessing import Preprocessing

class SegmentationDataset(Dataset):

    def __init__(self, images, masks, class_num=None, transform=None, save_masks=False):
        self.images = images
        self.masks = masks
        self.class_num = class_num
        self.transform = transform
        self.save_masks = save_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        p = Preprocessing()
        mask = p.mask_thresholding(mask, self.class_num, index)
        if self.save_masks:
            self.save_mask(mask, self.class_num, index)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = image.permute(2, 0, 1)
        mask = torch.Tensor(mask.copy())[None, :, :]

        return image, mask

    @classmethod
    def save_mask(cls, mask, class_num, index):
        mask_files = [file_names for file_names in glob.glob('Thesis/Data/Maps/vLayers/*.tiff')]
        mask_files.sort()
        mask = mask.cpu().detach().numpy()
        os.makedirs(f'/Users/celine/Desktop/Thesis/Data/Dataset/masks_{class_num}_classes', exist_ok=True)
        mask.imsave(f'/Users/celine/Desktop/Thesis/Data/Dataset/masks_{class_num}_classes/' + mask_files[index][25:-5]
                    + '.tiff', mask[:, :, 0], format='tiff', cmap='gray')
