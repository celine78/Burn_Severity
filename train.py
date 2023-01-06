import time
import subprocess
import numpy as np
from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from dataset import SegmentationDataset
from preprocessing import Preprocessing, Normalize
from augmentation import DataAugmentation, Compose
from metrics import mIoU, pixel_accuracy, get_lr, plot_loss, plot_acc, plot_score, predict_image_mask_miou, \
    miou_score, pixel_acc, predict, predict_image_pixel

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
    def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, patch=False):
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
                logger.debug(f'output size: {output.size()}')
                logger.debug(f'mask size: {mask.size()}')
                mask = mask[:, 0, :, :].long()
                logger.debug(f'mask size {mask.size()}')
                if mask.max() == 0:
                    plt.imshow(mask.squeeze())
                    print(mask)
                    break
                #plt.imshow(mask.squeeze())
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
                        mask = mask[:, 0, :, :].long()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path, mask_path = '/Users/celine/Desktop/Thesis/Data/Landsat_images_512/*/*/response.tiff', \
        '/Users/celine/Desktop/Thesis/Data/Maps/vLayers/*.tiff'
    class_num = 2
    logger.info('Loading dataset')
    images_dir, masks_dir = Preprocessing.load_dataset(image_path, mask_path)
    logger.info('Resizing images and masks')
    images, masks = map(list, zip(*[Preprocessing.resize(img_dir, mask_dir) for img_dir, mask_dir in zip(images_dir, masks_dir)]))
    logger.info('Data augmentation')
    images = [Preprocessing.delete_landsat_bands(image, [9, 12, 13]) for image in images]
    mean, std = Preprocessing.get_mean_std(images)
    vflip, hflip, rota, shear = DataAugmentation.data_augmentation(0.5, 0.5, 0.25, 0.25, 40, 20)
    #vflip, hflip, rota, shear = da.data_augmentation(1, 1, 1, 1, 40, 20)
    trans = Compose([
        vflip, hflip, rota, shear,
        Normalize(mean, std)
    ])
    logger.info('Create datasets')
    dataset = SegmentationDataset(images, masks, class_num=class_num, transform=trans)
    logger.debug(f'Dataset length: {len(dataset)}')
    train_dataset, val_dataset, test_dataset = Train.dataset_split(dataset, [0.7, 0.15, 0.15])
    logger.info(f'Train set: {len(train_dataset)} | Validation set: {len(val_dataset)} '
                f'| Test set: {len(test_dataset)}')

    model_id = 0
    if os.path.exists(f'/Users/celine/burn-severity/model_{model_id}.pt'):
        print(f'Load model with id {model_id}')
        model = torch.load(f'/Users/celine/burn-severity/model_{model_id}.pt')
    else:
        print(f'Model training')
        batch_size = 3
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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                        steps_per_epoch=int(len(train_dataset)/batch_size))
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

        history = Train.fit(epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, device)
        torch.save(model, f'/Users/celine/burn-severity/model_{model_id}.pt')

        plot_loss(history)
        plot_score(history)
        plot_acc(history)

    #image, mask = test_dataset[0]
    import skimage
    image = skimage.io.imread('/Users/celine/Downloads/response.tiff')
    image = Preprocessing.resize_image(image)
    image = Preprocessing.delete_landsat_bands(image, [9, 12, 13])
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    trans = transforms.Compose([
        transforms.Normalize(torch.Tensor([1166.1479, 990.1230, 860.5953, 818.1750, 2062.4404, 1656.0735,
                                           1043.4313, 830.8372, 270.5279, 269.8681]),
                             torch.Tensor([445.0345, 427.1914, 453.4088, 571.3936, 1052.1285, 1009.8184,
                                           764.5239, 492.1391, 86.5190, 86.3010]))])
    image = trans(image)

    mask_predicted = model.predict(image.unsqueeze(0))
    mask_predicted = mask_predicted.squeeze().permute(1, 2, 0)
    mask_predicted_1 = Preprocessing.mask_binary_thresholding_li(mask_predicted[:, :, 1])
    mask_predicted_2 = predict_image_pixel(model, image, device)

    data = np.array(mask_predicted[:, :, 1])
    mask_predicted_transformed = (data - np.min(data)) / (np.max(data) - np.min(data))

    mask_predicted_3 = Preprocessing.mask_binary_thresholding_li(mask_predicted_transformed)
    mask_predicted_transformed[mask_predicted_transformed < 0.5] = 0
    mask_predicted_transformed[mask_predicted_transformed >= 0.5] = 1

    fig, ax = plt.subplots(1, 8, figsize=(10, 10))
    ax[0].imshow(mask_predicted[:, :, 0], cmap='gray')
    ax[1].imshow(mask_predicted[:, :, 1], cmap='gray')
    ax[2].imshow(mask_predicted_1, cmap='gray')
    ax[3].imshow(mask_predicted_2.permute(1, 2, 0)[:, :, 0], cmap='gray') #
    ax[4].imshow(mask_predicted_2.permute(1, 2, 0)[:, :, 1], cmap='gray')
    ax[5].imshow(mask_predicted_transformed, cmap='gray')
    ax[6].imshow(image.permute(1, 2, 0)[:, :, 4])
    #ax[7].imshow(mask.permute(1, 2, 0))
    plt.show()

    #print(model.predict(test_dataset[0][0].unsqueeze(0)))
    #print(model.get_extra_state())
    #print(model.training)

    """

    predict(model, test_dataset[0], device)

    mob_miou = miou_score(model, test_dataset, device)

    mob_acc = pixel_acc(model, test_dataset, device)

    print('Test Set mIoU', np.mean(mob_miou))
    print('Test Set Pixel Accuracy', np.mean(mob_acc))

"""