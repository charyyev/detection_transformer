import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import weight_reduce_loss

def l1_loss(pred, target, mask, loss_weight):
    mask = mask.unsqueeze(1).expand_as(pred).float()
    loss = F.l1_loss(pred, target, reduction="mean")
    #loss = loss / (mask.sum() + 1e-4)
    return weight_reduce_loss(loss * loss_weight, mask, avg_factor=mask.sum() + 1e-4)