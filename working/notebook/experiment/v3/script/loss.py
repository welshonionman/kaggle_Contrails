import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, mode="binary", smooth=1e-7):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.mode = mode
        self.smooth = smooth

    def forward(self, outputs, targets):
        bce_loss = smp.losses.SoftBCEWithLogitsLoss()(outputs, targets)
        dice_loss = smp.losses.DiceLoss(mode=self.mode, smooth=self.smooth)(outputs, targets)

        loss = (self.weight * dice_loss) + ((1-self.weight) * bce_loss)
        return loss


class DiceWCELoss(nn.Module):
    def __init__(self, weight=0.5, mode="binary", smooth=1e-7):
        super(DiceWCELoss, self).__init__()
        self.weight = weight
        self.mode = mode
        self.smooth = smooth

    def forward(self, outputs, targets):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pos_weight = torch.Tensor([7.31]).to(device)
        wce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=pos_weight)(outputs, targets)

        dice_loss = smp.losses.DiceLoss(mode=self.mode, smooth=self.smooth)(outputs, targets)

        loss = (self.weight * dice_loss) + ((1-self.weight) * wce_loss)
        return loss


def get_lossfn(CFG, smooth=0, mode="binary"):
    lossfn = None
    if CFG.loss == 'DiceLoss':
        lossfn = smp.losses.DiceLoss(mode=mode, smooth=smooth)
    if CFG.loss == 'Dice_BCE_Loss':
        lossfn = DiceBCELoss()
    if CFG.loss == 'WCELoss':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pos_weight = torch.Tensor([7.31]).to(device)
        lossfn = smp.losses.SoftBCEWithLogitsLoss(pos_weight=pos_weight)
    if CFG.loss == 'Dice_WCE_Loss':
        lossfn = DiceWCELoss()

    return lossfn
