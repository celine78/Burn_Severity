from dataset import SegmentationDataset
import torch.nn as nn
import time
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import torch
import numpy as np
from preprocessing import Preprocessing, Normalize, DeleteLandsatBands
from augmentation import DataAugmentation, Compose
import matplotlib.pyplot as plt

from metrics import mIoU, pixel_accuracy, get_lr


class Train(object):

    def dataset_split(self, dataset, p_splits: list):
        train_size = int(p_splits[0] * len(dataset))
        val_size = int(p_splits[1] * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        train, val, test = random_split(dataset, [train_size, val_size, test_size])
        return train, val, test

    def fit(self, epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
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
        for e in range(epochs):
            since = time.time()
            running_loss = 0
            iou_score = 0
            accuracy = 0
            # training loop
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
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
                mask = mask[0, :, :, :].long()
                if mask.max() == 0:
                    plt.imshow(mask.squeeze())
                    print(mask)
                    break
                plt.imshow(mask.squeeze())
                loss = criterion(output, mask)
                # evaluation metrics
                iou_score += mIoU(output, mask)
                accuracy += pixel_accuracy(output, mask)
                # backward
                loss.backward()
                optimizer.step()  # update weight
                optimizer.zero_grad()  # reset gradient

                # step the learning rate
                lrs.append(get_lr(optimizer))
                scheduler.step()

                running_loss += loss.item()

            else:
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
                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(val_loader))

                if min_loss > (test_loss / len(val_loader)):
                    print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                    min_loss = (test_loss / len(val_loader))
                    decrease += 1
                    if decrease % 5 == 0:
                        print('saving model...')
                        torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

                if (test_loss / len(val_loader)) > min_loss:
                    not_improve += 1
                    min_loss = (test_loss / len(val_loader))
                    print(f'Loss Not Decrease for {not_improve} time')
                    if not_improve == 7:
                        print('Loss not decrease for 7 times, Stop Training')
                        break

                # iou
                val_iou.append(val_iou_score / len(val_loader))
                train_iou.append(iou_score / len(train_loader))
                train_acc.append(accuracy / len(train_loader))
                val_acc.append(test_accuracy / len(val_loader))
                print("Epoch:{}/{}..".format(e + 1, epochs),
                      "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                      "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                      "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                      "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                      "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                      "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                      "Time: {:.2f}m".format((time.time() - since) / 60))

        history = {'train_loss': train_losses, 'val_loss': test_losses,
                   'train_miou': train_iou, 'val_miou': val_iou,
                   'train_acc': train_acc, 'val_acc': val_acc,
                   'lrs': lrs}
        print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
        return history


if __name__ == '__main__':
    image_path, mask_path = '/Users/celine/Desktop/Thesis/Data/Landsat_images_512/*/*/response.tiff', \
        '/Users/celine/Desktop/Thesis/Data/Maps/vLayers/*.tiff'
    train = Train()
    da = DataAugmentation()
    p = Preprocessing()
    class_num = 2
    vflip, hflip, rota, shear = da.data_augmentation(0.5, 0.5, 0.25, 0.25, 40, 20)
    images_dir, masks_dir = p.load_dataset(image_path, mask_path)
    images, masks = map(list, zip(*[p.resize(img_dir, mask_dir) for img_dir, mask_dir in zip(images_dir, masks_dir)]))
    mean, std = p.get_mean_std(images)

    trans = Compose([
        DeleteLandsatBands([9, 12, 13]),
        vflip, hflip, rota, shear,
        Normalize(mean, std),
    ])

    dataset = SegmentationDataset(images, masks, class_num=class_num, transform=trans)
    train_dataset, val_dataset, test_dataset = train.dataset_split(dataset, [0.7, 0.15, 0.15])
    batch_size = 1
    num_workers = 0
    pin_memory = True
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)

    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=4, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16], in_channels=10)
    max_lr = 1e-3
    epoch = 5
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

    history = train.fit(epoch, model, train_loader, val_loader, criterion, optimizer, scheduler)