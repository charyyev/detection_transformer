import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
import json
import torch
import torch.nn.functional as F

from core.datasets.dataset import Dataset
from utils.utils import voxel_to_points

class Vis():
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0
        self.canvas = SceneCanvas(keys='interactive',
                                show=True,
                                size=(1600, 900))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene,
                                                    camera=TurntableCamera(distance=30.0))
        self.grid.add_widget(self.scan_view)
        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)
        self.bbox = vispy.scene.visuals.Line(parent=self.scan_view.scene)

        self.canvas1 = SceneCanvas(keys='interactive',
                                show=True,
                                size=(400, 400))
        self.canvas1.events.key_press.connect(self._key_press)
        self.canvas1.events.draw.connect(self._draw)

        self.view = self.canvas1.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        

        self.update_scan()


    def get3D_corners(self,  bbox):
        h, w, l, x, y, z, yaw = bbox[1:]

        corners = []
        front = l / 2
        back = -l / 2
        left = w / 2
        right = -w / 2
        top = h
        bottom = 0
        corners.append([front, left, top])
        corners.append([front, left, bottom])
        corners.append([front, right, bottom])
        corners.append([front, right, top])
        corners.append([back, left, top])
        corners.append([back, left, bottom])
        corners.append([back, right, bottom])
        corners.append([back, right, top])
        
        for i in range(8):
            corners[i] = self.rotate_pointZ(corners[i], yaw)[0] + np.array([x, y, z])
       
        return corners

    def rotate_pointZ(self, point, yaw):
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])

        rotated_point = np.matmul(rotation_matrix, np.reshape(point, (3, 1)))
        return np.reshape(rotated_point, (1, 3))

    def plot_boxes(self, cls_list, boxes):
        object_colors = {0: np.array([1, 0, 0, 1]), 
                         1: np.array([0, 1, 0, 1]), 
                         2: np.array([0, 0, 1, 1]),
                         3: np.array([1, 1, 0, 1]),
                         4: np.array([1, 1, 1, 1])}
        connect = []
        points = []
        colors = []

        for i, box in enumerate(boxes):
            color = np.tile(object_colors[cls_list[i]], (8, 1))
            corners = np.array(box)
            j = 8 * i
            con = [[j, j + 1],
                   [j + 1, j + 2],
                   [j + 2, j + 3],
                   [j + 3, j],
                   [j + 4, j + 5],
                   [j + 5, j + 6],
                   [j + 6, j + 7],
                   [j + 7, j + 4],
                   [j, j + 4],
                   [j + 1, j + 5],
                   [j + 2, j + 6],
                   [j + 3, j + 7]]
            con = np.array(con)

            if i == 0:
                points = corners
                connect = con
                colors = color
            else:
                points = np.concatenate((points, corners), axis = 0)
                connect = np.concatenate((connect, con), axis = 0)
                colors = np.concatenate((colors, color), axis = 0)

        self.bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)


    def get_point_color_using_intensity(self, points):
        scale_factor = 500
        scaled_intensity = np.clip(points[:, 3] * scale_factor, 0, 255)
        scaled_intensity = scaled_intensity.astype(np.uint8)
        cmap = plt.get_cmap("viridis")

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0
        colors = color_range[scaled_intensity]
        return colors

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid], indexing = "xy")
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base


    def update_scan(self):
        data = self.dataset[self.index]
        voxel = data["voxel"].permute(2, 1, 0).numpy()
        boxes = data["boxes"]
        heatmap = data["heatmap"].unsqueeze(0)
        data_type = data["data_type"]

        nms_kernel_size = 3
        padding = nms_kernel_size // 2
        batch_size = 1
        num_proposals = 128
        geometry = dataset.config[data_type]["geometry"]

        x_size = 400
        y_size = 400
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)

        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        
        ## for Pedestrian
        local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
        
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.contiguous().view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :num_proposals]
        top_scores, _ = heatmap.view(batch_size, -1).sort(dim=-1, descending=True)

        top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='trunc')
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_labels = top_proposals_class[0]

        # add category embedding
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)[0]
        #print(query_pos)
        points = voxel_to_points(voxel, dataset.config[data_type]["geometry"])

        colors = np.array([0, 1, 1])
        
        self.canvas.title = str(self.index) + ": " + data_type
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)
        corners = []
        cls_list = []

        for i in range(query_pos.shape[0]):
            pos = query_pos[i]
            label = query_labels[i]
            center_x = pos[0]  * geometry["x_res"] + geometry["x_min"]
            center_y = pos[1]  * geometry["y_res"] + geometry["y_min"]
            box = [1, 1, 0.5, 0.5, center_x, center_y, 0, 0]
            corners.append(self.get3D_corners(box))
            cls_list.append(int(label))
        self.plot_boxes(cls_list, corners)
        
        
        img = np.max(data["heatmap"].numpy(), 0)
        #img = data["heatmap"][1].numpy()
        #print(img[img > 0])
        #img1 = np.copy(img)
        #img1[reg_mask == 1] = 1
        #img = np.concatenate((img, img1), axis = 0)
        self.image.set_data(img)



    def _key_press(self, event):
        if event.key == 'N':
            if self.index < len(self.dataset) - 1:
                self.index += 1
            self.update_scan()

        if event.key == 'B':
            if self.index > 0:
                self.index -= 1
            self.update_scan()

        if event.key == 'Q':
            self.destroy()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        self.canvas.app.run()


if __name__ == "__main__":
    data_file = "/home/stpc/clean_data/list/overfit1.txt"
    
    with open("/home/stpc/proj/detection_transformer/configs/base.json", 'r') as f:
        config = json.load(f)

    dataset = Dataset(data_file, config["data"], config["augmentation"])

    vis = Vis(dataset)
    vis.run()
