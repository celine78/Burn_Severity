# In this class, the model is trained and evaluated. This class contains two methods.
# In the first method, the dataset is split into a training set, an evaluation set and a test set. The ratio for the
# splitting is determined in the configuration found in the configuration file configurations.ini. A seed parameter for
# the validation and the test sets are determined to ensure reproducibility. A value of -1 is used in this case to
# indicate that the random split should not be deterministic.
# The second method fits the data to the model. The training uses pytorch and is evaluated using the methods in the
# metrics.py file. The results are logged in wandb, if enabled. An optimizer, a scheduler and a criterion can be given
# as parameters. An early stopping is possible, if enabled in the configuration.

import time
import numpy as np
import torch
from typing import Tuple

from matplotlib import pyplot as plt
from torch.utils.data import random_split, Subset
import configparser
from typing import Dict, List

from data.dataset import SegmentationDataset
from training.metrics import mIoU, pixel_accuracy, get_lr
from torch.utils.data import DataLoader

import logging.config
import wandb

logger = logging.getLogger(__name__)


class Train(object):
    """
    Train the model with a given training dataset
    """
    def __init__(self) -> None:
        """
        Constructor for the training class
        """
        self.config = configparser.ConfigParser()
        self.config.read('configurations.ini')

    def dataset_split(self, dataset: SegmentationDataset, ratio: List) -> Tuple[Subset, Subset, Subset]:
        """
        Splits the dataset into training, validation and test sets
        :param dataset: dataset to split
        :param ratio: ratio for each set
        :return: training, validation and test subsets
        """
        train_size = int(ratio[0] * len(dataset))
        val_size = int(ratio[1] * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        test_seed = self.config.getint('TRAIN', 'testSeed')
        val_seed = self.config.getint('TRAIN', 'valSeed')
        if test_seed != -1:
            generator = torch.manual_seed(test_seed)
            train_val, test = random_split(dataset, [(train_size + val_size), test_size], generator=generator)
        else:
            train_val, test = random_split(dataset, [(train_size + val_size), test_size])
        if val_seed != -1:
            generator = torch.manual_seed(test_seed)
            train, val = random_split(train_val, [train_size, val_size], generator=generator)
        else:
            train, val = random_split(train_val, [train_size, val_size])
        return train, val, test

    def fit(self, model, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer, device, scheduler,
            model_name) -> Dict:
        """
        Fit a model using the given training and validation sets
        :param model: model to be trained
        :param train_loader: training set
        :param val_loader: validation set
        :param criterion: criterion to be used
        :param optimizer: optimizer to use
        :param device: cpu or cuda device
        :param scheduler: scheduler to use
        :param model_name: name of the model
        :return: training history
        """
        train_losses = []
        test_losses = []
        val_iou = []
        val_acc = []
        train_iou = []
        train_acc = []
        lowest_loss = 1
        lrs = []
        min_loss = np.inf
        decrease = 1
        no_improvement = 0

        model.to(device)
        start_time = time.time()
        epochs = self.config.getint('TRAIN', 'epochs')
        logger.debug(f'Range epochs : {range(epochs)}')
        for epoch in range(epochs):
            logger.debug(f'Epoch e : {epoch}')
            since = time.time()
            running_loss = 0
            iou_score = 0
            accuracy = 0
            model.train()
            logger.info(f'train_loader: {train_loader}')
            # Batch training
            for data in train_loader:
                logger.debug(f'data length: {len(data)}')
                img, msk = data
                image = img.to(device)
                mask = msk.to(device)
                # forward
                output = model(image)
                logger.debug(f'output size: {output.size()}')
                logger.debug(f'mask size: {mask.size()}')
                mask = mask[:, 0, :, :].long()
                logger.debug(f'mask size {mask.size()}')
                loss = criterion(output, mask)
                logger.debug(f'loss: {loss}')
                # evaluation metrics
                iou_score += mIoU(output, mask, n_classes=self.config.getint("TRAIN", "classes_n"))
                logger.debug(f'iou_score: {iou_score}')
                accuracy += pixel_accuracy(output, mask)
                logger.debug(f'accuracy: {accuracy}')
                # backward
                loss.backward()
                optimizer.step()  # update weight
                optimizer.zero_grad()  # reset gradient

                # step the learning rate
                lrs.append(get_lr(optimizer))
                if scheduler is not None:
                    scheduler.step()
                    print(f'Learning rate in epoch {epoch}: ', scheduler.get_last_lr())
                    logger.debug(f'Learning rate in epoch {epoch}: {scheduler.get_last_lr()}')
                running_loss += loss.item()
            else:
                model.eval()
                val_loss = 0
                val_accuracy = 0
                val_iou_score = 0
                with torch.no_grad():
                    for data in val_loader:
                        img, msk = data
                        image = img.to(device)
                        mask = msk.to(device)
                        output = model(image)
                        # evaluation metrics
                        val_iou_score += mIoU(output, mask)
                        val_accuracy += pixel_accuracy(output, mask)
                        # loss
                        mask = mask[:, 0, :, :].long()
                        loss = criterion(output, mask)
                        val_loss += loss.item()

                # calculation mean for each batch
                # print('train loader length: ', len(train_loader))
                logger.info(f'train loader length: {len(train_loader)}')
                train_losses.append(running_loss / len(train_loader))
                test_losses.append(val_loss / len(val_loader))

                if min_loss > (val_loss / len(val_loader)):
                    # print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss / len(val_loader))))
                    logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {(val_loss / len(val_loader)):.3f}')
                    min_loss = (val_loss / len(val_loader))
                    decrease += 1

                if self.config.getboolean('TRAIN', 'saveModel') and val_loss < lowest_loss:
                    lowest_loss = val_loss
                    logger.info(f'Saving model')
                    # print('Saving model')
                    torch.save(model.state_dict(),
                               f'models/model_{model_name}_{self.config.getint("TRAIN", "classes_n")}_best.pt')

                val_iou.append(val_iou_score / len(val_loader))
                train_iou.append(iou_score / len(train_loader))
                train_acc.append(accuracy / len(train_loader))
                val_acc.append(val_accuracy / len(val_loader))

                if self.config.getboolean('WANDB', 'wandbLog'):
                    wandb.log({"epoch": epoch + 1,
                               "train loss": running_loss / len(train_loader),
                               "val loss": val_loss / len(val_loader),
                               "train mIoU": iou_score / len(train_loader),
                               "val mIoU": val_iou_score / len(val_loader),
                               "train accuracy": accuracy / len(train_loader),
                               "val accuracy": val_accuracy / len(val_loader),
                               })

                print("Epoch:{}/{}..".format(epoch + 1, epochs),
                      "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                      "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
                      "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                      "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                      "Train Accuracy:{:.3f}..".format(accuracy / len(train_loader)),
                      "Val Accuracy:{:.3f}..".format(val_accuracy / len(val_loader)),
                      "Time: {:.2f}m".format((time.time() - since) / 60))
                logger.info(f'Epoch: {epoch + 1}/{epochs} '
                            f'Train Loss: {(running_loss / len(train_loader)):.3f}..'
                            f'Val Loss: {(val_loss / len(val_loader)):.3f}..'
                            f'Train mIoU: {(iou_score / len(train_loader)):.3f}..'
                            f'Val mIoU: {(val_iou_score / len(val_loader)):.3f}..'
                            f'Train Accuracy: {(accuracy / len(train_loader)):.3f}..'
                            f'Val Accuracy: {(val_accuracy / len(val_loader)):.3f}..'
                            f'Time: {((time.time() - since) / 60):.2f}m'
                            )

                if self.config.getboolean('TRAIN', 'early_stop') and (val_loss / len(val_loader)) > min_loss:
                    no_improvement += 1
                    min_loss = (val_loss / len(val_loader))
                    print(f'Validation loss did not decreased for {no_improvement} time')
                    logger.info(f'Validation loss did not decreased for {no_improvement} time')
                    early_stop = 10
                    if no_improvement == early_stop:
                        print(f'Training stopped. Validation loss did not decrease for the last {early_stop} times.')
                        logger.info(f'Training stopped. No validation loss in the last {early_stop} times.')
                        break

        history = {'train_loss': train_losses, 'val_loss': test_losses,
                   'train_mIoU': train_iou, 'val_mIoU': val_iou,
                   'train_acc': train_acc, 'val_acc': val_acc,
                   'lrs': lrs}

        print('Total time: {:.2f} m'.format((time.time() - start_time) / 60))
        logger.info(f'Total time: {((time.time() - start_time) / 60):.2f} m')

        return history
