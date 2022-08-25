import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import weight_reduce_loss

def l1_loss(pred, target, mask, loss_weight):
    mask = mask.unsqueeze(1).expand_as(pred).float()

    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss * loss_weight