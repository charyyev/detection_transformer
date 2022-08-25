import torch
import torch.nn.functional as F
import numpy as np

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    
    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


def voxel_to_points(voxel, geometry):
    x_min = geometry["x_min"]
    x_max = geometry["x_max"]
    y_min = geometry["y_min"]
    y_max = geometry["y_max"]
    z_min = geometry["z_min"]
    z_max = geometry["z_max"]
    x_res = geometry["x_res"]
    y_res = geometry["y_res"]
    z_res = geometry["z_res"]

    xs, ys, zs = np.where(voxel.astype(int) == 1)
    points_x = xs + x_res / 2
    points_y = ys + y_res / 2
    points_z = zs + z_res / 2

    points_x = points_x * x_res + x_min
    points_y = points_y * y_res + y_min
    points_z = points_z * z_res + z_min
    #centers = np.array(np.where(voxel.astype(int) == 1)) + np.array([[x_res / 2], [y_res / 2], [z_res / 2]])
    return np.transpose(np.array([points_x, points_y, points_z]))