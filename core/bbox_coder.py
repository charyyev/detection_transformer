import torch
import json
from core.datasets.dataset import Dataset
from torch.utils.data import DataLoader

class BBoxCoder():
    def __init__(self,
                 cfg
                 ):

        self.cfg = cfg
        self.out_size_factor = cfg["test_cfg"]["out_size_factor"]


    def convert_format(self, dst_boxes):
        """ converts bboxes format

        Args:
            dst_boxes: Nx8 (cls, h, w, l, x, y, z, yaw)
            data_type (string): type of dataset
        Returns:
            torch.Tensor: Nx5 tensor of targets (x, y, w, l, yaw)
        """

        targets = torch.zeros([dst_boxes.shape[0], 5])
        targets[:, 0] = dst_boxes[:, 4]
        targets[:, 1] = dst_boxes[:, 5]
        
        targets[:, 2] = dst_boxes[:, 1]
        targets[:, 3] = dst_boxes[:, 2]
        targets[:, 4] = dst_boxes[:, 7]
       
        return targets, dst_boxes[:, 0].type(torch.long)


    def encode(self, dst_boxes, data_type):
        """ Encode bboxes

        Args:
            dst_boxes: Nx8 (cls, h, w, l, x, y, z, yaw)
            data_type (string): type of dataset
        Returns:
            torch.Tensor: Nx6 tensor of targets (x_feat, y_feat, log(w), log(l), cos(yaw), sin(yaw))
        """

        geometry = self.cfg[data_type]["geometry"]
        targets = torch.zeros([dst_boxes.shape[0], 6])
        targets[:, 0] = (dst_boxes[:, 4] - geometry["x_min"]) / (self.out_size_factor * geometry["x_res"])
        targets[:, 1] = (dst_boxes[:, 5] - geometry["y_min"]) / (self.out_size_factor * geometry["y_res"])
        
        targets[:, 2] = dst_boxes[:, 1].log()
        targets[:, 3] = dst_boxes[:, 2].log()
        targets[:, 4] = torch.sin(dst_boxes[:, 7])
        targets[:, 5] = torch.cos(dst_boxes[:, 7])
       
        return targets

    def decode(self, heatmap, rot, dim, center, data_type):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with the shape of
                [B, 2, num_proposals].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 2, num_proposals].
            center (torch.Tensor): bev center of the boxes with the shape of
                [B, 2, num_proposals]. (in feature map metric)
            

        Returns:
            list[dict]: Decoded boxes.
        """
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        geometry = self.cfg[data_type]["geometry"]

        # change size to real world metric
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * geometry["x_res"] + geometry["x_min"]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * geometry["y_res"] + geometry["y_min"]

        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()

        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        
        final_box_preds = torch.cat([center, dim, rot], dim=1).permute(0, 2, 1)


        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'boxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

if __name__ == "__main__":
    with open("/home/stpc/proj/detection_transformer/configs/base.json", 'r') as f:
            config = json.load(f)

    data_file = "/home/stpc/clean_data/list/train.txt"

    dataset = Dataset(data_file, config["data"], config["augmentation"])
    data_loader = DataLoader(dataset, shuffle=False, batch_size=4, collate_fn = dataset.collate_fn)
    coder = BBoxCoder(config["data"])
    for data in data_loader:
        voxel = data["voxel"]
        boxes = data["boxes"]
        targets = coder.encode(boxes[0], data["data_type"][0])
        num_proposals = targets.shape[0]
        heatmap = torch.zeros((num_proposals, 4))
        
        for i in range(4):
            heatmap[:, i][boxes[0][:, 0].type(torch.long) == i] = 1
        
        center = targets[:, 0:2].permute(1, 0).unsqueeze(0)
        dim = targets[:, 2: 4].permute(1, 0).unsqueeze(0)
        rot = targets[:, 4:6].permute(1, 0).unsqueeze(0)
        print(boxes[0])
        print(coder.decode(heatmap.permute(1, 0).unsqueeze(0), rot, dim, center, data["data_type"][0])[0])




        break