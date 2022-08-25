import torch
from utils.iou_calculator import compute_iou
from scipy.optimize import linear_sum_assignment

class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12
         binary_input (bool, optional): Whether the input is binary,
            default False.

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits
                in shape (num_query, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits.
            gt_labels (Tensor)): Labels.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)


class BBoxBEVL1Cost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, boxes, gt_boxes, geometry):
        pc_start = boxes.new([geometry['x_min'], geometry["y_min"]])
        pc_range = boxes.new([geometry["x_max"], geometry["y_max"]]) - pc_start
        
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (boxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_boxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight

class IoU3DCost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        iou_cost = - iou
        return iou_cost * self.weight


class HungarianAssigner():
    def __init__(self):
        self.cls_cost = FocalLossCost(weight = 0.15, alpha = 0.25, gamma = 2)
        self.reg_cost = BBoxBEVL1Cost(weight = 0.25)
        self.iou_cost = IoU3DCost(weight = 0.25)


    def assign(self, boxes, gt_boxes, gt_labels, cls_pred, geometry):
        # compute the weighted costs
        # see mmdetection/mmdet/core/bbox/match_costs/match_cost.py

        if gt_labels.shape[0] == 0:
            return [], []

        cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)
        reg_cost = self.reg_cost(boxes, gt_boxes, geometry)
        iou = compute_iou(boxes.detach().cpu().numpy(), gt_boxes.detach().cpu().numpy())
        iou_cost = self.iou_cost(torch.from_numpy(iou))

        # weighted sum of above three costs
        cost = cls_cost.detach().cpu() + reg_cost.detach().cpu() + iou_cost

        # do Hungarian matching on CPU using linear_sum_assignment
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(boxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(boxes.device)

        return matched_row_inds, matched_col_inds
        


if __name__ == "__main__":
    boxes = torch.tensor([[0, 0, 1, 1, 0],
                          [1, 0, 1, 1, 0],
                          [2, 0, 1, 1, 0],
                          [3, 0, 1, 1, 0]])

    cls_pred = torch.tensor([[1, 0], [0, 1], [0, 1], [0,1]]).T.unsqueeze(0)

    gt_boxes = torch.tensor([[0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 0 ]])
                             
    gt_labels = torch.tensor([1, 0])
    geometry = {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}

    assigner = HungarianAssigner()
    matched_rows, matched_cols = assigner.assign(boxes, gt_boxes, gt_labels, cls_pred, geometry)
    print(matched_rows)
    print(matched_cols)

    
    
    