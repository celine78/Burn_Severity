import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from dataset import SegmentationDataset
from preprocessing import Preprocessing, Normalize
from augmentation import DataAugmentation, Compose
from metrics import mIoU, pixel_accuracy, get_lr

import logging.config
import wandb

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)


"""
WANDB_API_KEY = '750ade35afcb71cb11253e643f3ed1509fbd4ddf'
subprocess.check_output(['wandb', 'login', WANDB_API_KEY, '--relogin'])
wandb.init(project="burn-severity")
"""


class Train(object):

    @staticmethod
    def dataset_split(dataset, p_splits: list):
        train_size = int(p_splits[0] * len(dataset))
        val_size = int(p_splits[1] * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        train, val, test = random_split(dataset, [train_size, val_size, test_size])
        return train, val, test

    @staticmethod
    def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
        torch.cuda.empty_cache()
        train_losses = []
        test_losses = []
        val_iou = []
        val_acc = []
        train_iou = []
        train_acc = []
        lrs = []
        min_loss = np.inf
        decrease = 1
        not_improve = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        fit_time = time.time()
        logger.debug(f'Range epochs : {range(epochs)}')
        for e in range(epochs):
            logger.debug(f'Epoch e : {e}')
            since = time.time()
            running_loss = 0
            iou_score = 0
            accuracy = 0
            # training loop
            model.train()
            logger.debug(f'tqdm(train_loader): {tqdm(train_loader)}')
            for i, data in enumerate(tqdm(train_loader)):
                logger.debug(f'data length: {len(data)}')
                # training phase
                image_tiles, mask_tiles = data
                if patch:
                    bs, n_tiles, c, h, w = image_tiles.size()

                    image_tiles = image_tiles.view(-1, c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)

                image = image_tiles.to(device)
                mask = mask_tiles.to(device)
                # forward
                output = model(image)
                # logger.debug(f'output size: {output.size()}')
                mask = mask[0, :, :, :].long()
                # logger.debug(f'mask size {mask.size()}')
                if mask.max() == 0:
                    plt.imshow(mask.squeeze())
                    print(mask)
                    break
                plt.imshow(mask.squeeze())
                loss = criterion(output, mask)
                logger.debug(f'loss: {loss}')
                # evaluation metrics
                iou_score += mIoU(output, mask)
                logger.debug(f'iou_score: {iou_score}')
                accuracy += pixel_accuracy(output, mask)
                logger.debug(f'accuracy: {accuracy}')
                # backward
                loss.backward()
                optimizer.step()  # update weight
                optimizer.zero_grad()  # reset gradient

                # step the learning rate
                lrs.append(get_lr(optimizer))
                scheduler.step()
                running_loss += loss.item()
            else:
                # logger.debug(f'model.eval(): {model.eval()}')
                model.eval()
                test_loss = 0
                test_accuracy = 0
                val_iou_score = 0
                # validation loop
                with torch.no_grad():
                    for i, data in enumerate(tqdm(val_loader)):
                        # reshape to 9 patches from single image, delete batch size
                        image_tiles, mask_tiles = data

                        if patch:
                            bs, n_tiles, c, h, w = image_tiles.size()

                            image_tiles = image_tiles.view(-1, c, h, w)
                            mask_tiles = mask_tiles.view(-1, h, w)

                        image = image_tiles.to(device)
                        mask = mask_tiles.to(device)
                        output = model(image)
                        # evaluation metrics
                        val_iou_score += mIoU(output, mask)
                        test_accuracy += pixel_accuracy(output, mask)
                        # loss
                        mask = mask[0, :, :, :].long()
                        loss = criterion(output, mask)
                        test_loss += loss.item()

                # calculation mean for each batch
                print('train loader length: ', len(train_loader))
                logger.info(f'train loader length: {len(train_loader)}')
                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(val_loader))

                if min_loss > (test_loss / len(val_loader)):
                    print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                    logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {(test_loss / len(val_loader)):.3f}')
                    min_loss = (test_loss / len(val_loader))
                    decrease += 1
                    if decrease % 5 == 0:
                        print('saving model...')
                        # torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

                if (test_loss / len(val_loader)) > min_loss:
                    not_improve += 1
                    min_loss = (test_loss / len(val_loader))
                    print(f'Loss Not Decrease for {not_improve} time')
                    logger.info(f'Loss Not Decrease for {not_improve} time')
                    if not_improve == 7:
                        print('Loss not decrease for 7 times, Stop Training')
                        logger.info('Loss not decrease for 7 times, Stop Training')
                        break

                # iou
                val_iou.append(val_iou_score / len(val_loader))
                train_iou.append(iou_score / len(train_loader))
                train_acc.append(accuracy / len(train_loader))
                val_acc.append(test_accuracy / len(val_loader))
                """
                wandb.log({"epoch": e + 1,
                           "train loss": running_loss / len(train_loader),
                           "val loss": test_loss / len(val_loader),
                           "train mIoU": iou_score / len(train_loader),
                           "val mIoU": val_iou_score / len(val_loader),
                           "train accuracy": accuracy / len(train_loader),
                           "val accuracy": test_accuracy / len(val_loader),
                           })
                """
                print("Epoch:{}/{}..".format(e + 1, epochs),
                      "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                      "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                      "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                      "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                      "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                      "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                      "Time: {:.2f}m".format((time.time() - since) / 60))
                logger.info(f'Epoch: {e + 1}/{epochs} '
                            f'Train Loss: {(running_loss / len(train_loader)):.3f}..'
                            f'Val Loss: {(test_loss / len(val_loader)):.3f}..'
                            f'Train mIoU: {(iou_score / len(train_loader)):.3f}..'
                            f'Val mIoU: {(val_iou_score / len(val_loader)):.3f}..'
                            f'Train Acc: {(accuracy / len(train_loader)):.3f}..'
                            f'Val Acc: {(test_accuracy / len(val_loader)):.3f}..'
                            f'Time: {((time.time() - since) / 60):.2f}m'
                            )

        history = {'train_loss': train_losses, 'val_loss': test_losses,
                   'train_miou': train_iou, 'val_miou': val_iou,
                   'train_acc': train_acc, 'val_acc': val_acc,
                   'lrs': lrs}
        print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
        logger.info(f'Total time: {((time.time() - fit_time) / 60):.2f} m')
        return history


if __name__ == '__main__':
    image_path, mask_path = '/Users/celine/Desktop/Thesis/Data/Landsat_images_512/*/*/response.tiff', \
        '/Users/celine/Desktop/Thesis/Data/Maps/vLayers/*.tiff'
    train = Train()
    da = DataAugmentation()
    p = Preprocessing()
    class_num = 2
    logger.info('Loading dataset')
    images_dir, masks_dir = p.load_dataset(image_path, mask_path)
    logger.info('Resizing images and masks')
    images, masks = map(list, zip(*[p.resize(img_dir, mask_dir) for img_dir, mask_dir in zip(images_dir, masks_dir)]))
    logger.info('Data augmentation')
    images = [p.delete_landsat_bands(image, [9, 12, 13]) for image in images]
    mean, std = p.get_mean_std(images)
    vflip, hflip, rota, shear = da.data_augmentation(0.5, 0.5, 0.25, 0.25, 40, 20)
    #vflip, hflip, rota, shear = da.data_augmentation(1, 1, 1, 1, 40, 20)
    trans = Compose([
        vflip, hflip, rota, shear,
        Normalize(mean, std)
    ])
    logger.info('Create datasets')
    dataset = SegmentationDataset(images, masks, class_num=class_num, transform=trans)
    logger.debug(f'Dataset length: {len(dataset)}')
    train_dataset, val_dataset, test_dataset = train.dataset_split(dataset, [0.7, 0.15, 0.15])
    logger.info(f'Train set: {len(train_dataset)} | Validation set: {len(val_dataset)} '
                f'| Test set: {len(test_dataset)}')
    batch_size = 1
    num_workers = 0
    pin_memory = True
    logger.info('Create dataloaders')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)
    logger.info('Load architecture with backbone')

    backbone = 'resnet34'
    encoder_weights = 'imagenet'
    encoder_depth = 5
    decoder_channels = [256, 128, 64, 32, 16]
    in_channels = 10

    model = smp.Unet(backbone, encoder_weights=encoder_weights, classes=class_num, activation=None,
                     encoder_depth=encoder_depth,
                     decoder_channels=decoder_channels, in_channels=in_channels)
    max_lr = 1e-3
    epoch = 5
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))
    logger.info('Train model')

    """
    wandb.config.class_num = class_num
    wandb.config.epochs = epoch
    wandb.config.batch_size = batch_size
    wandb.config.in_channels = in_channels
    wandb.config.backbone = backbone
    wandb.config.encoder_weights = encoder_weights
    wandb.config.encoder_depth = encoder_depth
    wandb.config.decoder_channels = decoder_channels
    wandb.config.weight_decay = weight_decay
    wandb.config.max_lr = max_lr
    """

    history = train.fit(epoch, model, train_loader, val_loader, criterion, optimizer, scheduler)
