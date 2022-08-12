import torch

class BBoxCoder():
    def __init__(self,
                 cfg
                 ):

        self.cfg = cfg
        self.out_size_factor = cfg["out_size_factor"]


    def encode(self, dst_boxes, data_type):
        """ Encode bboxes

        Args:
            dst_boxes: Nx8 (cls, h, w, l, x, y, z, yaw)
            data_type (string): type of dataset
        Returns:
            torch.Tensor: Nx6 tensor of targets (x_feat, y_feat, log(h), log(w), cos(yaw), sin(yaw))
        """
        geometry = self.cfg[data_type]
        targets = torch.zeros([dst_boxes.shape[0], 7]).to(dst_boxes.device)
        targets[:, 0] = (dst_boxes[:, 4] - geometry["x_min"]) / (self.out_size_factor * geometry["x_size"])
        targets[:, 1] = (dst_boxes[:, 5] - geometry["y_min"]) / (self.out_size_factor * geometry["y_size"])
        
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

        geometry = self.cfg[data_type]

        # change size to real world metric
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * geometry["x_size"] + geometry["x_min"]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * geometry["y_size"] + geometry["y_min"]

        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()

        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        
        final_box_preds = torch.cat([center, rot, dim], dim=1).permute(0, 2, 1)


        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts
