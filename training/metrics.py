# This class implements the metrics for the evaluation of the training and evaluation of the model. The methods include
# the scores for accuracy and mean over intersection (mIoU). A further method gets the learning rate from the optimizer.
# Four methods can be used to plot the accuracy, the mIoU, the loss and the prediction.


from typing import Dict, Tuple, List

import numpy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Subset

import logging.config

logger = logging.getLogger(__name__)


def pixel_accuracy(output: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute the pixel accuracy of the predicted output
    :param output: predicted output from the model
    :param mask: ground truth mask
    :return: accuracy
    """
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-10, n_classes: int = 2) -> numpy.ndarray:
    """
    Compute the mean intersection over union score of the predicted output
    :param pred_mask: predicted output from the model
    :param mask: ground truth mask
    :param smooth: smooth factor
    :param n_classes: number of classes
    :return: intersection over union mean
    """
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_lr(optimizer):
    """
    Get the learning rate of the optimizer
    :param optimizer: optimizer
    :return: learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def predict(model, image: torch.Tensor, device: torch.device, plot: bool = False) -> None:
    """
    Predict the output
    :param model: the trained model
    :param image: image to predict
    :param device: device to use
    :param plot: whether to plot the result
    """
    img, mask = image
    pred_mask, score = predict_image_mask_pixel(model, img, mask, device)
    if plot:
        plot_predictions(img, mask, pred_mask, score)


def predict_image_mask_miou(model, image: torch.Tensor, mask: torch.Tensor, device: torch.device) -> \
        Tuple[torch.Tensor, numpy.ndarray]:
    """
    Get the predicted mask of an image with its mean intersection over union score
    :param model: trained model
    :param image: image to predict
    :param mask: ground truth mask
    :param device: device to use
    :return: predicted mask and mIoU score
    """
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def predict_image_pixel(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Predict the image pixel-wise
    :param model: trained model
    :param image: image to predict
    :param device: device to use
    :return: predicted mask
    """
    model.eval()
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def predict_image_mask_pixel(model, image: torch.Tensor, mask: torch.Tensor, device: torch.device) -> \
        Tuple[torch.Tensor, float]:
    """
    Predict an image with its ground truth mask
    :param model: trained model
    :param image: image to predict
    :param mask: ground truth mask
    :param device: device to use
    :return: predicted mask and its accuracy score
    """
    model.eval()
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score(model, test_set: Subset, device: torch.device) -> List[numpy.ndarray]:
    """
    Get the mean intersection over union score over a test set
    :param model: trained model
    :param test_set: test set
    :param device: device to use
    :return: mean intersection over union score
    """
    score_iou = []
    for i in range(len(test_set)):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask, device)
        score_iou.append(score)
    return score_iou


def pixel_acc(model, test_set: Subset, device: torch.device) -> List[float]:
    """
    Get the accuracy score over a test set
    :param model: trained model
    :param test_set: test set
    :param device: device to use
    :return: accuracy score
    """
    accuracy = []
    for i in range(len(test_set)):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask, device)
        accuracy.append(acc)
    return accuracy

def plot_loss(history: Dict) -> None:
    """
    Plot the loss function
    :param history: the training history
    """
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_mIoU(history: Dict) -> None:
    """
    Plot the mean intersection of union
    :param history: the training history
    """
    plt.plot(history['train_mIoU'], label='train_mIoU', marker='*')
    plt.plot(history['val_mIoU'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history: Dict) -> None:
    """
    Plot the accuracy
    :param history: the training history
    """
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_predictions(image: torch.Tensor, mask: torch.Tensor, pred_mask: torch.Tensor, score: float) -> None:
    """
    Plot the image with the predicted mask and the ground truth
    :param image: image predicted
    :param mask: ground truth mask
    :param pred_mask: predicted mask
    :param score: accuracy score
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(image[4, :, :])
    ax[0].set_title('Picture')

    ax[1].imshow(mask.squeeze())
    ax[1].set_title('Ground truth')
    ax[1].set_axis_off()

    ax[2].imshow(pred_mask.squeeze())
    ax[2].set_title('UNet-Burn Severity | accuracy {:.3f}'.format(score))
    ax[2].set_axis_off()
    plt.show()