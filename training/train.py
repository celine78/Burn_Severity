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
from typing import Tuple, Any
from torch.utils.data import random_split, Subset
import configparser
from typing import List
from tqdm import tqdm
from datetime import datetime

from data.dataset import SegmentationDataset
from training.metrics import get_lr
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Accuracy, AveragePrecision, Dice, F1Score, Precision, Recall, ROC, AUROC

import logging.config
import wandb

logger = logging.getLogger('burn_severity')


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
            model_name) -> tuple[dict, Any]:
        """
        Fit a model using the given training and validation sets
        :param model: model to train
        :param train_loader: training set
        :param val_loader: validation set
        :param criterion: criterion to be used
        :param optimizer: optimizer to use
        :param device: cpu or cuda device
        :param scheduler: scheduler to use
        :param model_name: name of the model
        :return: training history
        """
        train_acc = []
        train_iou = []
        train_ap = []
        train_dice = []
        train_f1 = []
        train_precision = []
        train_recall = []
        train_auroc = []

        val_acc = []
        val_iou = []
        val_ap = []
        val_dice = []
        val_f1 = []
        val_precision = []
        val_recall = []
        val_auroc = []

        train_losses = []
        val_losses = []
        lowest_loss = np.inf
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
            start = time.time()
            train_loss = 0
            train_iou_score = 0
            train_acc_score = 0
            train_ap_score = 0
            train_dice_score = 0
            train_f1_score = 0
            train_precision_score = 0
            train_recall_score = 0
            train_auroc_score = 0
            number_classes = self.config.getint("TRAIN", "classes_n")
            task = "binary" if number_classes == 2 else "multiclass"
            jaccard = JaccardIndex(task=task, num_classes=number_classes)
            accuracy = Accuracy(task=task, num_classes=number_classes)
            ap = AveragePrecision(task=task, num_classes=number_classes)
            dice = Dice(num_classes=number_classes)
            f1 = F1Score(task=task, num_classes=number_classes)
            precision = Precision(task=task, num_classes=number_classes)
            recall = Recall(task=task, num_classes=number_classes)
            auroc = AUROC(task=task, num_classes=number_classes)
            model.train()
            # Batch training
            for data in tqdm(train_loader):
                logger.debug(f'data length: {len(data)}')
                img, msk = data
                image = img.to(device)
                mask = msk.to(device)
                # forward
                output = model(image)
                logger.debug(f'output size: {output.size()}')
                logger.debug(f'mask size: {mask.size()}')
                # dimension reduction for loss computation
                mask = mask[:, 0, :, :].long()
                logger.debug(f'mask size {mask.size()}')
                loss = criterion(output, mask)
                logger.debug(f'loss: {loss}')
                # evaluation metrics
                prediction = torch.argmax(output, dim=1)
                train_iou_score += jaccard(prediction, mask)
                train_acc_score += accuracy(prediction, mask)
                if number_classes == 2:
                    preds = torch.amax(output, dim=1)
                    train_ap_score += ap(preds, mask)
                    train_auroc_score += auroc(preds, mask)
                else:
                    train_ap_score += ap(output, mask)
                    train_auroc_score += auroc(output, mask)

                train_dice_score += dice(prediction, mask)
                train_f1_score += f1(prediction, mask)
                train_precision_score += precision(prediction, mask)
                train_recall_score += recall(prediction, mask)

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
                train_loss += loss.item()
            else:
                model.eval()
                val_loss = 0
                val_acc_score = 0
                val_iou_score = 0
                val_ap_score = 0
                val_dice_score = 0
                val_f1_score = 0
                val_precision_score = 0
                val_recall_score = 0
                val_auroc_score = 0

                with torch.no_grad():
                    for data in tqdm(val_loader):
                        img, msk = data
                        image = img.to(device)
                        mask = msk.to(device)
                        output = model(image)
                        mask = mask[:, 0, :, :].long()
                        prediction = torch.argmax(output, dim=1)
                        val_iou_score += jaccard(prediction, mask)
                        val_acc_score += accuracy(prediction, mask)
                        if number_classes == 2:
                            preds = torch.amax(output, dim=1)
                            val_ap_score += ap(preds, mask)
                            val_auroc_score += auroc(preds, mask)
                        else:
                            val_ap_score += ap(output, mask)
                            val_auroc_score += auroc(output, mask)

                        val_dice_score += dice(prediction, mask)
                        val_f1_score += f1(prediction, mask)
                        val_precision_score += precision(prediction, mask)
                        val_recall_score += recall(prediction, mask)

                        loss = criterion(output, mask)
                        val_loss += loss.item()

                logger.info(f'train loader length: {len(train_loader)}')
                train_loss_mean = train_loss / len(train_loader)
                val_loss_mean = val_loss / len(val_loader)
                train_losses.append(train_loss_mean)
                val_losses.append(val_loss_mean)

                if min_loss > val_loss_mean:
                    # print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, val_loss_mean))
                    logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {val_loss_mean:.3f}')
                    min_loss = val_loss_mean
                    decrease += 1

                if self.config.getboolean('TRAIN', 'saveModel') and val_loss < lowest_loss:
                    lowest_loss = val_loss
                    logger.info(f'Saving model')
                    # print('Saving model')
                    torch.save(model,
                               f'models/Landsat_L2_model_'
                               f'2W_10k_{model_name}_{self.config.getint("TRAIN", "classes_n")}_'
                               f'{str(datetime.now().strftime("%d-%m-%Y-%H:%M"))}_best.pt')

                train_iou_mean = train_iou_score / len(train_loader)
                train_acc_mean = train_acc_score / len(train_loader)
                train_ap_mean = train_ap_score / len(train_loader)
                train_dice_mean = train_dice_score / len(train_loader)
                train_f1_mean = train_f1_score / len(train_loader)
                train_precision_mean = train_precision_score / len(train_loader)
                train_recall_mean = train_recall_score / len(train_loader)
                train_auroc_mean = train_auroc_score / len(train_loader)

                train_iou.append(train_iou_mean)
                train_acc.append(train_acc_mean)
                train_ap.append(train_ap_mean)
                train_dice.append(train_dice_mean)
                train_f1.append(train_f1_mean)
                train_precision.append(train_precision_mean)
                train_recall.append(train_recall_mean)
                train_auroc.append(train_auroc_mean)

                val_iou_mean = val_iou_score / len(val_loader)
                val_acc_mean = val_acc_score / len(val_loader)
                val_ap_mean = val_ap_score / len(val_loader)
                val_dice_mean = val_dice_score / len(val_loader)
                val_f1_mean = val_f1_score / len(val_loader)
                val_precision_mean = val_precision_score / len(val_loader)
                val_recall_mean = val_recall_score / len(val_loader)
                val_auroc_mean = val_auroc_score / len(val_loader)

                val_iou.append(val_iou_mean)
                val_acc.append(val_acc_mean)
                val_ap.append(val_ap_mean)
                val_dice.append(val_dice_mean)
                val_f1.append(val_f1_mean)
                val_precision.append(val_precision_mean)
                val_recall.append(val_recall_mean)
                val_auroc.append(val_auroc_mean)

                if self.config.getboolean('WANDB', 'wandbLog'):
                    wandb.log({"epoch": epoch + 1,
                               "train accuracy": train_acc_mean,
                               "train IoU": train_iou_mean,
                               "train average precision": train_ap_mean,
                               "train dice": train_dice_mean,
                               "train f1": train_f1_mean,
                               "train precision": train_precision_mean,
                               "train recall": train_recall_mean,
                               "train AUROC": train_auroc_mean,

                               "val accuracy": val_acc_mean,
                               "val mIoU": val_iou_mean,
                               "val average precision": val_ap_mean,
                               "val dice": val_dice_mean,
                               "val f1": val_f1_mean,
                               "val precision": val_precision_mean,
                               "val recall": val_recall_mean,
                               "val AUROC": val_auroc_mean,

                               "train loss": train_loss_mean,
                               "val loss": val_loss_mean,
                               })

                print(f'Epoch: {int(epoch + 1)} / {int(epochs)}',
                      f'Train accuracy: {train_acc_mean:.3f}..',
                      f'Train IoU: {train_iou_mean:.3f}..',
                      f'Train average precision: {train_ap_mean:.3f}..',
                      f'Train dice: {train_dice_mean:.3f}..',
                      f'Train f1: {train_f1_mean:.3f}..',
                      f'Train precision: {train_precision_mean:.3f}..',
                      f'Train recall: {train_recall_mean:.3f}..',
                      f'Train AUROC: {train_auroc_mean:.3f}..'

                      f'Val accuracy: {val_acc_mean:.3f}..',
                      f'Val IoU: {val_iou_mean:.3f}..',
                      f'Val average precision: {val_ap_mean:.3f}..',
                      f'Val dice: {val_dice_mean:.3f}..',
                      f'Val f1: {val_f1_mean:.3f}..',
                      f'Val precision: {val_precision_mean:.3f}..',
                      f'Val recall: {val_recall_mean:.3f}..',
                      f'Val AUROC: {val_auroc_mean:.3f}..',

                      f'Train loss: {train_loss_mean:.3f}..',
                      f'Val loss: {val_loss_mean:.3f}..',
                      f'Time: {((time.time() - start) / 60):.2f} minutes'
                      )
                """
                logger.info(f'Epoch: {int(epoch + 1)} / {int(epochs)}',
                            f'Train accuracy: {train_acc_mean:.3f}..',
                            f'Train IoU: {train_iou_mean:.3f}..',
                            f'Train average precision: {train_ap_mean:.3f}..',
                            f'Train dice: {train_dice_mean:.3f}..',
                            f'Train f1: {train_f1_mean:.3f}..',
                            f'Train precision: {train_precision_mean:.3f}..',
                            f'Train recall: {train_recall_mean:.3f}..',
                            f'Train AUROC: {train_auroc_mean:.3f}..'

                            f'Val accuracy: {val_acc_mean:.3f}..',
                            f'Val mIoU: {val_iou_mean:.3f}..',
                            f'Val average precision: {val_ap_mean:.3f}..',
                            f'Val dice: {val_dice_mean:.3f}..',
                            f'Val f1: {val_f1_mean:.3f}..',
                            f'Val precision: {val_precision_mean:.3f}..',
                            f'Val recall: {val_recall_mean:.3f}..',
                            f'Val ROC: {val_auroc_mean:.3f}..',

                            f'Train loss: {train_loss_mean:.3f}..',
                            f'Val loss: {val_loss_mean:.3f}..',
                            f'Time: {((time.time() - start) / 60):.2f} minutes'
                            )
                """

                if self.config.getboolean('TRAIN', 'early_stop') and val_loss_mean > min_loss:
                    no_improvement += 1
                    min_loss = val_loss_mean
                    print(f'Validation loss did not decreased for {no_improvement} time')
                    logger.info(f'Validation loss did not decreased for {no_improvement} time')
                    early_stop = 10
                    if no_improvement == early_stop:
                        print(f'Training stopped. Validation loss did not decrease for the last {early_stop} times.')
                        logger.info(f'Training stopped. No validation loss in the last {early_stop} times.')
                        break

        history = {'train_acc': train_acc,
                   'train_iou': train_iou,
                   'train_ap': train_ap,
                   'train_dice': train_dice,
                   'train_f1': train_f1,
                   'train_precision': train_precision,
                   'train_recall': train_recall,
                   'train_auroc': train_auroc,
                   'val_iou': val_iou,
                   'val_acc': val_acc,
                   'val_ap': val_ap,
                   'val_dice': val_dice,
                   'val_f1': val_f1,
                   'val_precision': val_precision,
                   'val_recall': val_recall,
                   'val_auroc': val_auroc,
                   'train_loss': train_losses,
                   'val_loss': val_losses,
                   'lrs': lrs
                   }

        print(f'Total time: {((time.time() - start_time) / 60):.2f} minutes')
        logger.info(f'Total time: {((time.time() - start_time) / 60):.2f} minutes')

        return history, model
