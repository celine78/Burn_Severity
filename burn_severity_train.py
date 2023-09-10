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
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex, Accuracy, AveragePrecision, Dice, F1Score, Precision, Recall, AUROC


from training.metrics import plot_loss, plot_acc, plot_mIoU
from unet.trans_unet_plus2.trans_unet_plus2 import transunet_plus2
from utils import Preprocessing, Normalize
from utils.augmentation import DataAugmentation, Compose
from data.dataset import SegmentationDataset
from training.train import Train
from unet.trans_unet.vit_seg_modeling import VisionTransformer as ViT_seg
from unet.trans_unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from unet.vanilla_unet.unet_arch import UNet
from self_attention_cv.transunet import TransUnet
from self_attention_cv import ViT, ResNet50ViT

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger('burn_severity')
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

    logger.info('Removing satellite bands')
    images = [prep.delete_landsat_bands(image, json.loads(config.get('DATA', 'deleteBands')), 'L2') for image in
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

    test_loader = DataLoader(test_set, shuffle=False, batch_size=config.getint('TRAIN', 'batch_size'),
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
        """
        vit = config.get('TRANSUNET', 'vit')
        config_vit = CONFIGS_ViT_seg[vit]
        config_vit.n_classes = config.getint('TRAIN', 'classes_n')
        config_vit.n_skip = config.getint('TRANSUNET', 'skip_n')
        model = ViT_seg(config_vit, img_size=config.getint('TRAIN', 'input_size'),
                        num_classes=config_vit.n_classes).to(device)
        model.load_from(weights=np.load(config_vit.pretrained_path))
        model_name = config.get('TRANSUNET', 'name')

        model = TransUNet(image_size=256, pretrain=True, num_classes=config.getint('TRAIN', 'classes_n'),
                          decoder_channels=[256,128,64,16])
        """
        #model = transunet_plus2(input_size=(512, 512, 8), filter_num=[16, 32, 64, 128], n_labels=2)

        #resnetVit = ResNet50ViT(img_dim=512, pretrained_resnet=True, num_classes=2, resnet_layers=6)
        model_name = config.get('TRANSUNET', 'name')
        model = TransUnet(in_channels=8, img_dim=512, vit_blocks=12, vit_dim_linear_mhsa_block=1024, classes=2)

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
        wandb.config.daataset = config.get('DATA', 'imagesPath')
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
        #elif config.getboolean('TRANSUNET', 'use_model'):
            #wandb.config.vit = config.get('TRANSUNET', 'vit')
            #wandb.config.skip_num = config.getint('TRANSUNET', 'skip_n')
            #wandb.config.backbone = config.get('TRANSUNET', 'backbone')

    history, model = train.fit(model, train_loader, val_loader, criterion, optimizer, device, scheduler, model_name)

    number_classes = config.getint('TRAIN', 'classes_n')
    task = "binary" if number_classes == 2 else "multiclass"
    jaccard = JaccardIndex(task=task, num_classes=number_classes)
    accuracy = Accuracy(task=task, num_classes=number_classes)
    ap = AveragePrecision(task=task, num_classes=number_classes)
    dice = Dice(num_classes=number_classes)
    f1 = F1Score(task=task, num_classes=number_classes)
    precision = Precision(task=task, num_classes=number_classes)
    recall = Recall(task=task, num_classes=number_classes)
    auroc = AUROC(task=task, num_classes=number_classes)

    # cm = ConfusionMatrix(task=task, num_classes=number_classes)
    # model = model.load_state_dict(
    #    torch.load(f'/models/model_{model_name}_{config.getint("TRAIN", "classes_n")}_best.pt'))

    test_loss = 0
    test_iou_score = 0
    test_acc_score = 0
    test_ap_score = 0
    test_dice_score = 0
    test_f1_score = 0
    test_precision_score = 0
    test_recall_score = 0
    test_auroc_score = 0

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img, msk = data
            image = img.to(device)
            mask = msk.to(device)
            mask = mask[:, 0, :, :].long()
            output = model(image)
            pred = torch.argmax(output, dim=1)

            test_iou_score += jaccard(pred, mask)
            test_acc_score += accuracy(pred, mask)
            test_dice_score += dice(pred, mask)
            test_f1_score += f1(pred, mask)
            test_precision_score += precision(pred, mask)
            test_recall_score += recall(pred, mask)
            if number_classes == 2:
                preds = torch.amax(output, dim=1)
                test_ap_score += ap(preds, mask)
                test_auroc_score += auroc(preds, mask)
            else:
                test_ap_score += ap(output, mask)
                test_auroc_score += auroc(output, mask)

            loss = criterion(output, mask)
            test_loss += loss.item()

    test_iou_mean = test_iou_score / len(test_loader)
    test_acc_mean = test_acc_score / len(test_loader)
    test_ap_mean = test_ap_score / len(test_loader)
    test_dice_mean = test_dice_score / len(test_loader)
    test_f1_mean = test_f1_score / len(test_loader)
    test_precision_mean = test_precision_score / len(test_loader)
    test_recall_mean = test_recall_score / len(test_loader)
    test_auroc_mean = test_auroc_score / len(test_loader)

    if config.getboolean('WANDB', 'wandbLog'):
        wandb.log({"test accuracy": test_acc_mean,
                   "test IoU": test_iou_mean,
                   "test average precision": test_ap_mean,
                   "test dice": test_dice_mean,
                   "test f1": test_f1_mean,
                   "test precision": test_precision_mean,
                   "test recall": test_recall_mean,
                   "test AUROC": test_auroc_mean,
                   })

    logger.info(f'Test IoU: {test_iou_mean}')
    logger.info(f'Test accuracy: {test_acc_mean}')
    logger.info(f'Test average precision: {test_ap_mean}')
    logger.info(f'Test dice: {test_dice_mean}')
    logger.info(f'Test f1: {test_f1_mean}')
    logger.info(f'Test precision: {test_precision_mean}')
    logger.info(f'Test recall: {test_recall_mean}')
    logger.info(f'Test AUROC: {test_auroc_mean}')
    print('Test IoU', test_iou_mean)
    print('Test accuracy', test_acc_mean)
    print('Test average precision', test_ap_mean)
    print('Test dice', test_dice_mean)
    print('Test f1', test_f1_mean)
    print('Test precision', test_precision_mean)
    print('Test recall', test_recall_mean)
    print('Test AUROC', test_auroc_mean)
    return history


if __name__ == '__main__':
    history = base_train()
    config = configparser.ConfigParser()
    config.read('configurations.ini')
    if config.getboolean('WANDB', 'wandbLog') is False:
        plot_loss(history)
        plot_acc(history)
        plot_mIoU(history)
