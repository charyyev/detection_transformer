import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import json

from core.torchplus import Sequential, Empty, change_default_args
from core.models.transfusion_head import TransFusionHead
from core.datasets.dataset import Dataset
from core.bbox_coder import BBoxCoder
from core.assigner import HungarianAssigner
from core.losses.focal_loss import FocalLoss
from core.losses.general_losses import l1_loss
from core.losses.gaussian_focal_loss import GaussianFocalLoss
from utils.utils import clip_sigmoid


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 layer_nums=(8, 8),
                 layer_strides=(1, 2),
                 num_filters=(32, 64),
                 upsample_strides=(1, 2),
                 num_upsample_filters=(64, 64),
                 num_input_features=35):

        super(RPN, self).__init__()

        assert len(layer_nums) == 2
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        

    def forward(self, x):

        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        
        x = torch.cat([up1, up2], dim=1)
        return x



class TransFusion(nn.Module):
    def __init__(self, cfg):
        super(TransFusion, self).__init__()
        self.backbone = RPN()
        common_heads=dict(center=(2, 2), dim=(2, 2), rot=(2, 2))
        self.header = TransFusionHead(num_classes = cfg["num_classes"], common_heads = common_heads, test_cfg=cfg["test_cfg"])

    def forward(self, x):
        features = self.backbone(x)
        pred = self.header(features)

        return pred


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super(SetCriterion, self).__init__()
        self.box_coder = BBoxCoder(cfg)
        self.assigner = HungarianAssigner()
        
        self.cls_loss_fn = FocalLoss(use_sigmoid=True, reduction="mean", loss_weight=1.0)
        self.heatmap_loss_fn = GaussianFocalLoss(reduction="mean", loss_weight=1.0)
        
        self.cfg = cfg
        self.num_classes = cfg["num_classes"]



    def forward(self, pred, gt_boxes, heatmap, data_types):
        label_targets, box_targets, masks = self.get_targets(pred, gt_boxes, data_types)
        cls_loss = self.cls_loss_fn(pred[0]["heatmap"], label_targets)
        heatmap_loss = self.heatmap_loss_fn(clip_sigmoid(pred[0]["dense_heatmap"]), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
        center_loss = l1_loss(pred[0]["center"], box_targets[:, 0:2, :], masks, loss_weight=0.25)
        dim_loss = l1_loss(pred[0]["center"], box_targets[:, 2:4, :], masks, loss_weight=0.25)
        rot_loss = l1_loss(pred[0]["center"], box_targets[:, 4:6, :], masks, loss_weight=0.25)

        loss = cls_loss + heatmap_loss + center_loss + dim_loss + rot_loss
        loss_dict = {"loss": loss, 
                     "cls": cls_loss.item(), 
                     "heatmap": heatmap_loss.item(), 
                     "center": center_loss.item(), 
                     "dim": dim_loss.item(), 
                     "rot": rot_loss.item()}

        return loss_dict
        

    def get_targets(self, pred, gt_boxes, data_types):
        list_of_pred_dict = []
        for batch_idx in range(len(gt_boxes)):
            pred_dict = {}
            for key in pred[0].keys():
                pred_dict[key] = pred[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)
    
        assert len(gt_boxes) == len(list_of_pred_dict)


        for i in range(len(gt_boxes)):
            label, box_target, mask = self.get_targets_single(list_of_pred_dict[i], gt_boxes[i], data_types[i])
            if i == 0:
                labels = label.unsqueeze(0)
                box_targets = box_target.unsqueeze(0)
                masks = mask.unsqueeze(0)
            else:
                labels = torch.cat((labels, label.unsqueeze(0)), dim = 0)
                box_targets = torch.cat((box_targets, box_target.unsqueeze(0)), dim = 0)
                masks = torch.cat((masks, mask.unsqueeze(0)), dim = 0)
        
        return labels, box_targets.permute(0, 2, 1), masks
            


    def get_targets_single(self, pred, gt_boxes, data_type):
        num_proposals = pred['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(pred['heatmap'].detach())
        center = copy.deepcopy(pred['center'].detach())
        dim = copy.deepcopy(pred['dim'].detach())
        rot = copy.deepcopy(pred['rot'].detach())
        
        boxes_dict = self.box_coder.decode(score, rot, dim, center, data_type)  # decode the prediction to real world metric bbox
        gt_boxes, gt_labels = self.box_coder.convert_format(gt_boxes)
        boxes = boxes_dict[0]['boxes'].to(score.device)

        assigned_rows, assigned_cols = self.assigner.assign(boxes, gt_boxes, gt_labels, score, self.cfg[data_type]["geometry"])


        # create target for loss computation
        box_targets = torch.zeros([num_proposals, 6]).to(center.device)
        labels = boxes.new_zeros(num_proposals, dtype=torch.long)
        mask = boxes.new_zeros(num_proposals, dtype = torch.long)
        
        labels += self.num_classes


        #both pos and neg have classification loss, only pos has regression and iou loss
        if len(assigned_rows) > 0:
            box_targets[assigned_rows] = self.box_coder.encode(gt_boxes, data_type)[assigned_cols]
            labels[assigned_rows] = gt_labels[assigned_cols]
            mask[assigned_rows] = 1

        return labels, box_targets, mask


if __name__ == "__main__":
    with open("/home/stpc/proj/detection_transformer/configs/base.json", 'r') as f:
        config = json.load(f)

    data_file = "/home/stpc/clean_data/list/train.txt"

    model = TransFusion(config["data"])
    criterion = SetCriterion(config["data"])
    dataset = Dataset(data_file, config["data"], config["augmentation"])
    data_loader = DataLoader(dataset, shuffle=False, batch_size=4, collate_fn = dataset.collate_fn)
    for data in data_loader:
        voxel = data["voxel"]
        boxes = data["boxes"]
        data_types = data["data_type"]
        heatmap = data["heatmap"]
        pred = model(voxel)
        criterion(pred, boxes, heatmap, data_types)

        break

    
