import torch.nn as nn
import numpy as np
import torch


class Dice(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(Dice, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, inputs, targets, smooth=1):
        if self.use_sigmoid:
            inputs = self.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return dice


def calc_dice_score(pred, true, thresh: float) -> float:
    dice = Dice(use_sigmoid=False)
    pred_thresh = np.where(pred > thresh, 1, 0)
    pred_thresh = torch.flatten(torch.from_numpy(pred_thresh))
    return dice(true, pred_thresh).item()


def calc_optim_thresh(pred, true, threshs_to_test):
    best_dice = -1
    for thresh in threshs_to_test:
        dice = calc_dice_score(pred, true, thresh)
        if dice > best_dice:
            best_dice = dice
            best_thresh = thresh
    return best_dice, best_thresh
