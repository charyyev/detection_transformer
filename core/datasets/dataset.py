import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
import math
import json


from utils.transform import Random_Rotation, Random_Scaling, OneOf, Random_Translation
from utils.gaussian import gaussian_radius, draw_heatmap_gaussian

from torch.utils.data import DataLoader


class Dataset(Dataset):
    def __init__(self, data_file, config, aug_config, task = "train") -> None:
        self.data_file = data_file

        self.create_data_list()

        self.config = config
        self.task = task
        self.num_classes = self.config["num_classes"]
        self.transforms = self.get_transforms(aug_config)
        self.augment = OneOf(self.transforms, aug_config["p"])
       

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file = '{}.bin'.format(self.data_list[idx])
        data_type = self.data_type_list[idx]

        pointcloud_folder = os.path.join(self.config[data_type]["location"], "pointcloud")
        lidar_path = os.path.join(pointcloud_folder, file)

        points = self.read_points(lidar_path)

        if self.task == "test":
            scan = self.voxelize(points, self.config[data_type]["geometry"])
            scan = torch.from_numpy(scan)
            scan = scan.permute(2, 0, 1)
            return {"voxel": scan,
                    "points": points,
                    "dtype": data_type
                }

        boxes = self.get_boxes(idx)

        if self.task == "train" and boxes.shape[0] != 0:
            points, boxes[:, 1:] = self.augment(points, boxes[:, 1:8])

        boxes = torch.from_numpy(boxes)
        heatmap = self.create_heatmap(boxes, data_type)

        scan = self.voxelize(points, self.config[data_type]["geometry"])
        scan = torch.from_numpy(scan)
        scan = scan.permute(2, 0, 1)


        if self.task == "val":
            class_list, boxes = self.read_bbox(boxes)
    
            return {"voxel": scan,
                    "boxes": boxes, 
                    "cls_list": class_list,
                    "points": points,
                    "boxes": boxes,
                    "heatmap": heatmap,
                    "dtype": data_type
                }   

        return {"voxel": scan, 
                "boxes": boxes,
                "data_type": data_type,
                "heatmap": heatmap
            }

            

    def read_points(self, lidar_path):
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    def voxelize(self, points, geometry):
        x_min = geometry["x_min"]
        x_max = geometry["x_max"]
        y_min = geometry["y_min"]
        y_max = geometry["y_max"]
        z_min = geometry["z_min"]
        z_max = geometry["z_max"]
        x_res = geometry["x_res"]
        y_res = geometry["y_res"]
        z_res = geometry["z_res"]

        x_size = int((x_max - x_min) / x_res)
        y_size = int((y_max - y_min) / y_res)
        z_size = int((z_max - z_min) / z_res)

        eps = 0.001

        #clip points
        x_indexes = np.logical_and(points[:, 0] > x_min + eps, points[:, 0] < x_max - eps)
        y_indexes = np.logical_and(points[:, 1] > y_min + eps, points[:, 1] < y_max - eps)
        z_indexes = np.logical_and(points[:, 2] > z_min + eps, points[:, 2] < z_max - eps)
        pts = points[np.logical_and(np.logical_and(x_indexes, y_indexes), z_indexes)]

        occupancy_mask = np.zeros((pts.shape[0], 3), dtype = np.int32)
        voxels = np.zeros((x_size, y_size, z_size), dtype = np.float32)
        occupancy_mask[:, 0] = (pts[:, 0] - x_min) // x_res
        occupancy_mask[:, 1] = (pts[:, 1] - y_min) // y_res
        occupancy_mask[:, 2] = (pts[:, 2] - z_min) // z_res

        idxs = np.array([occupancy_mask[:, 0].reshape(-1), occupancy_mask[:, 1].reshape(-1), occupancy_mask[:, 2].reshape( -1)])

        voxels[idxs[0], idxs[1], idxs[2]] = 1
        return np.swapaxes(voxels, 0, 1)


    def get_boxes(self, idx):
        '''
        :param i: the ith velodyne scan in the train/val set
        : return boxes of shape N:8

        '''

        f_name = '{}.txt'.format(self.data_list[idx])
        data_type = self.data_type_list[idx]
        label_folder = os.path.join(self.config[data_type]["location"], "label")
        label_path = os.path.join(label_folder, f_name)
        object_list = self.config[data_type]["objects"]
        boxes = []

        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    boxes.append(bbox)

        return np.array(boxes)

    def create_heatmap(self, boxes, data_type):
        geometry = self.config[data_type]["geometry"]
        grid_size = torch.tensor(self.config['test_cfg']['grid_size'])
        out_size_factor = self.config["test_cfg"]["out_size_factor"]
        
        feature_map_size = torch.div(grid_size[:2], out_size_factor, rounding_mode='trunc')  # [x_len, y_len]
        heatmap = boxes.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(boxes.shape[0]):
            width = boxes[idx][2]
            length = boxes[idx][3]
            width = width / geometry["x_res"] / out_size_factor
            length = length / geometry["y_res"] / out_size_factor
            if width > 0 and length > 0:
                radius = gaussian_radius((length, width), min_overlap=self.config['gaussian_overlap'])
                radius = max(self.config['min_radius'], int(radius))
                x, y = boxes[idx][4], boxes[idx][5]

                coor_x = (x - geometry["x_min"]) / geometry["x_res"] / out_size_factor
                coor_y = (y - geometry["y_min"]) / geometry["y_res"] / out_size_factor

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32)
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(heatmap[int(boxes[idx][0])], center_int, radius)

        return heatmap




    def read_bbox(self, boxes):
        corner_list = []
        class_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            class_list.append(box[0])
            corners = self.get3D_corners(box)
            corner_list.append(corners)
        return (class_list, corner_list)


    def get_transforms(self, config):
        transforms = []
        if config["rotation"]["use"]:
            limit_angle = config["rotation"]["limit_angle"]
            p = config["rotation"]["p"]
            transforms.append(Random_Rotation(limit_angle, p))
        if config["scaling"]["use"]:
            range = config["scaling"]["range"]
            p = config["scaling"]["p"]
            transforms.append(Random_Scaling(range, p))

        if config["translation"]["use"]:
            scale = config["translation"]["scale"]
            p = config["translation"]["p"]
            transforms.append(Random_Translation(scale, p))

        return transforms

    def create_data_list(self):
        data_list = []
        data_type_list = []
        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                data, data_type = line.split(";")
                data_list.append(data)
                data_type_list.append(data_type)
        
        self.data_list = data_list
        self.data_type_list = data_type_list


    def collate_fn(self, batch):
        boxes = []
        data_types = []
        voxels = []
        heatmaps = []
        
        for data in batch:
            boxes.append(data["boxes"])
            data_types.append(data["data_type"])
            voxels.append(data["voxel"].unsqueeze(0))
            heatmaps.append(data["heatmap"].unsqueeze(0))

        return {
            "voxel": torch.cat(voxels),
            "boxes": boxes,
            "heatmap": torch.cat(heatmaps),
            "data_type": data_types
        }


if __name__ == "__main__":
    data_file = "/home/stpc/clean_data/list/train.txt"

    with open("/home/stpc/proj/detection_transformer/configs/base.json", 'r') as f:
        config = json.load(f)

    dataset = Dataset(data_file, config["data"], config["augmentation"])
    data_loader = DataLoader(dataset, shuffle=False, batch_size=4, collate_fn = dataset.collate_fn)

    for data in data_loader:
        boxes = data["boxes"]
        voxel = data["voxel"]
        heatmap = data["heatmap"]
        print(voxel.shape)
        print(heatmap[heatmap > 0])
        #voxel = voxel.permute(1, 2, 0)
        #print(voxel.shape)
        #print(torch.sum(voxel, axis = 2))
        #print(label[:, 0].shape)
        #imgplot = plt.imshow(label[:, :, 0])
        #plt.imshow(torch.sum(voxel, axis = 2), cmap="brg", vmin=0, vmax=255)
        #img = voxel_to_img(voxel)
        #imgplot = plt.imshow(img)
       # plt.show()
        break