# This python file is the main entry point for the training of the model. The base train function takes care of the
# initialization before loading, preprocessing, augmenting and splitting the dataset. The model is also chosen and its
# hyperparameters are set. All the configurations can be set in the configurations.ini file. After the training, the
# model is being evaluated on the test set. If wandb is not enabled, the training results are being plot.

import wandb
import torch
import json
import subprocess
import logging.config
import configparser
import numpy as np
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from training.metrics import miou_score, pixel_acc, plot_loss, plot_acc, plot_mIoU
from utils import Preprocessing, Normalize
from utils.augmentation import DataAugmentation, Compose
from data.dataset import SegmentationDataset
from training.train import Train
from unet.trans_unet.vit_seg_modeling import VisionTransformer as ViT_seg
from unet.trans_unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unet.vanilla_unet.unet_arch import UNet

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)


def base_train() -> Dict:
    """
    Base training with hyperparameters definition, dataset preprocessing and training of a model
    :return: training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = configparser.ConfigParser()
    config.read('configurations.ini')

    if config.getboolean('WANDB', 'wandbLog'):
        WANDB_API_KEY = config.get('WANDB', 'wandbKey')
        subprocess.check_output(['wandb', 'login', WANDB_API_KEY, '--relogin'])
        wandb.init(project="burn-severity")

    logger.info('Loading dataset')
    prep = Preprocessing()
    images_dir, masks_dir = prep.load_dataset(config.get('DATA', 'imagesPath'), config.get('DATA', 'masksPath'))
    logger.info('Resizing images and masks')
    images, masks = map(list, zip(*[prep.resize(img_dir, mask_dir) for img_dir, mask_dir in
                                    zip(images_dir, masks_dir)]))

    if config.getboolean('PREPROCESSING', 'tiling'):
        logger.info('Thresholding masks')
        masks = [prep.mask_thresholding(mask, config.getint('TRAIN', 'classes_n'), i) for i, mask in enumerate(masks)]
        logger.info('Tiling images and masks')
        images, masks = map(list, zip(*[prep.image_tiling(img, msk, 256, 0.2) for img, msk in zip(images, masks)]))
        images = prep.flatten_list(images)
        masks = prep.flatten_list(masks)
        if config.getboolean('PREPROCESSING', 'filter_tiles'):
            logger.info('Filtering tiles')
            images, masks = prep.filter_masks(images, masks)

    logger.info('Removing Satellite bands')
    images = [prep.delete_landsat_bands(image, json.loads(config.get('DATA', 'deleteBands'))) for image in
              images]

    logger.info('Data augmentation init')

    mean, std = prep.get_mean_std(images)
    # min, max = Preprocessing.get_min_max(images)
    vflip, hflip, rota, shear = DataAugmentation.data_augmentation(
        config.getfloat('DATA AUGMENTATION', 'hflipProba'),
        config.getfloat('DATA AUGMENTATION', 'vflipProba'),
        config.getfloat('DATA AUGMENTATION', 'rotatonProba'),
        config.getfloat('DATA AUGMENTATION', 'shearProba'),
        config.getint('DATA AUGMENTATION', 'rotationAngle'),
        config.getint('DATA AUGMENTATION', 'shearAngle'),
        config.getboolean('DATA AUGMENTATION', 'randomRotation'),
        config.getboolean('DATA AUGMENTATION', 'randomShear'))

    trans = Compose([vflip, hflip, rota, shear, Normalize(mean=mean, std=std)])

    logger.info('Create datasets')

    dataset = SegmentationDataset(images, masks, class_num=config.getint('TRAIN', 'classes_n'), transform=trans,
                                  save_masks=config.getboolean('PREPROCESSING', 'saveMasks'))

    logger.debug(f'Dataset length: {len(dataset)}')

    train = Train()
    train_set, val_set, test_set = train.dataset_split(dataset, ratio=json.loads(
        config.get('TRAIN', 'trainValTestRatio')))

    logger.info(f'Train set: {len(train_set)} | Validation set: {len(val_set)} | Test set: {len(test_set)}')

    logger.info('Create dataloaders')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.getint('TRAIN', 'batch_size'),
                              num_workers=config.getint('TRAIN', 'num_workers'),
                              pin_memory=config.getboolean('TRAIN', 'pin_memory'))
    val_loader = DataLoader(val_set, shuffle=False, batch_size=config.getint('TRAIN', 'batch_size'),
                            num_workers=config.getint('TRAIN', 'num_workers'),
                            pin_memory=config.getboolean('TRAIN', 'pin_memory'))

    logger.info('Load model architecture')

    model_name = ''
    model = None
    criterion = None
    scheduler = None

    if config.getboolean('U-NET W BACKBONE', 'use_model'):
        model = smp.Unet(config.get('U-NET W BACKBONE', 'backbone'),
                         encoder_weights=config.get('U-NET W BACKBONE', 'encoder_weights'),
                         classes=config.getint('TRAIN', 'classes_n'), activation=None,
                         encoder_depth=config.getint('U-NET W BACKBONE', 'encoder_depth'),
                         decoder_channels=json.loads(config.get('U-NET W BACKBONE', 'decoder_channels')),
                         in_channels=config.getint('U-NET W BACKBONE', 'in_channels'))

        model_name = config.get('U-NET W BACKBONE', 'name')
        if config.getboolean('WANDB', 'wandbLog'):
            wandb.config.model = model_name

    elif config.getboolean('U-NET', 'use_model'):
        model = UNet(config.getint('U-NET', 'in_channels'), config.getint('TRAIN', 'classes_n'))
        model_name = config.get('U-NET', 'name')
        if config.getboolean('WANDB', 'wandbLog'):
            wandb.config.model = model_name

    elif config.getboolean('TRANSUNET', 'use_model'):
        vit = config.get('TRANSUNET', 'vit')
        config_vit = CONFIGS_ViT_seg[vit]
        config_vit.n_classes = config.getint('TRAIN', 'classes_n')
        config_vit.n_skip = config.getint('TRANSUNET', 'skip_n')
        # BEWARE: in the case of image tilling, the input_size needs to be modified
        model = ViT_seg(config_vit, img_size=config.getint('TRAIN', 'input_size'),
                        num_classes=config_vit.n_classes).to(device)
        model.load_from(weights=np.load(config_vit.pretrained_path))
        model_name = config.get('TRANSUNET', 'name')
        if config.getboolean('WANDB', 'wandbLog'):
            wandb.config.model = model_name

    if config.get('LOSS FUNCTION', 'name') == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.getfloat('OPTIMIZER', 'learning_rate'),
                                  weight_decay=config.getfloat('OPTIMIZER', 'weight_decay'))

    if config.getboolean('SCHEDULER', 'use_scheduler'):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=config.getfloat('OPTIMIZER', 'learning_rate'),
                                                        epochs=config.getint('TRAIN', 'epochs'),
                                                        steps_per_epoch=int(len(train_set) /
                                                                            config.getint('TRAIN', 'batch_size')) + 2)
        if config.getboolean('WANDB', 'wandbLog'):
            wandb.config.scheduler_name = config.getboolean('SCHEDULER', 'name')

    logger.info('Train model')

    if config.getboolean('WANDB', 'wandbLog'):
        wandb.config.class_num = config.getint('TRAIN', 'classes_n')
        wandb.config.epochs = config.getint('TRAIN', 'epochs')
        wandb.config.batch_size = config.getint('TRAIN', 'batch_size')
        wandb.config.weight_decay = config.getfloat('OPTIMIZER', 'weight_decay')
        wandb.config.max_lr = config.getfloat('OPTIMIZER', 'learning_rate')
        wandb.config.train_set_size = len(train_set)
        wandb.config.val_set_size = len(val_set)
        wandb.config.train_val_test_ratio = config.get('TRAIN', 'trainValTestRatio')
        wandb.config.scheduler = config.getboolean('SCHEDULER', 'use_scheduler')
        wandb.config.hflipProba = config.getfloat('DATA AUGMENTATION', 'hflipProba')
        wandb.config.vflipProba = config.getfloat('DATA AUGMENTATION', 'vflipProba')
        wandb.config.rotatonProba = config.getfloat('DATA AUGMENTATION', 'rotatonProba')
        wandb.config.shearProba = config.getfloat('DATA AUGMENTATION', 'shearProba')
        if config.getboolean('PREPROCESSING', 'tiling'):
            wandb.config.tiles_threshold = config.getfloat('PREPROCESSING', 'tiling_threshold')
        if config.getboolean('U-NET W BACKBONE', 'use_model'):
            wandb.config.backbone = config.get('U-NET W BACKBONE', 'backbone')
            wandb.config.encoder_weights = config.get('U-NET W BACKBONE', 'encoder_weights')
            wandb.config.encoder_depth = config.get('U-NET W BACKBONE', 'encoder_depth')
            wandb.config.decoder_channels = config.get('U-NET W BACKBONE', 'decoder_channels')
        elif config.getboolean('U-NET', 'use_model'):
            wandb.config.in_channels = config.getint('U-NET', 'in_channels')
        elif config.getboolean('TRANSUNET', 'use_model'):
            wandb.config.vit = config.get('TRANSUNET', 'vit')
            wandb.config.skip_num = config.getint('TRANSUNET', 'n_skip')
            wandb.config.backbone = config.get('TRANSUNET', 'backbone')

    history = train.fit(model, train_loader, val_loader, criterion, optimizer, device, scheduler, model_name)

    # model = model.load_state_dict(
    #    torch.load(f'/models/model_{model_name}_{config.getint("TRAIN", "classes_n")}_best.pt'))
    mob_mIoU = miou_score(model, test_set, device)
    mob_acc = pixel_acc(model, test_set, device)

    if config.getboolean('WANDB', 'wandbLog'):
        wandb.log({"Test mIoU": np.mean(mob_mIoU),
                   "Test accuracy": np.mean(mob_acc)
                   })

    logger.info(f'Test set mIoU: {np.mean(mob_mIoU)}')
    logger.info(f'Test set Pixel Accuracy: {np.mean(mob_acc)}')
    print('Test set mIoU', np.mean(mob_mIoU))
    print('Test set Pixel Accuracy', np.mean(mob_acc))
    return history


if __name__ == '__main__':
    history = base_train()
    config = configparser.ConfigParser()
    config.read('configurations.ini')
    if config.getboolean('WANDB', 'wandbLog') is False:
        plot_loss(history)
        plot_acc(history)
        plot_mIoU(history)
