
import numpy as np
from shapely.geometry import Polygon
import math


import torch.nn.functional as F

# input Nx5 (center, dim, rot)
def center_to_corners(boxes):
    center_x, center_y, w, l, yaw = np.split(boxes, 5, axis=1)

    cos_t = np.cos(yaw)
    sin_t = np.sin(yaw)

    rear_left_x = center_x - l/2 * cos_t - w/2 * sin_t
    rear_left_y = center_y - l/2 * sin_t + w/2 * cos_t
    rear_right_x = center_x - l/2 * cos_t + w/2 * sin_t
    rear_right_y = center_y - l/2 * sin_t - w/2 * cos_t
    front_right_x = center_x + l/2 * cos_t + w/2 * sin_t
    front_right_y = center_y + l/2 * sin_t - w/2 * cos_t
    front_left_x = center_x + l/2 * cos_t - w/2 * sin_t
    front_left_y = center_y + l/2 * sin_t + w/2 * cos_t

    corners = np.concatenate([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                               front_right_x, front_right_y, front_left_x, front_left_y], axis=1)

    corners = np.reshape(corners, (center_x.shape[0], 4, 2))

    return corners

def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(boxes_a, boxes_b):
    """Calculates IoU of the given box with the array of the given boxes.
    boxes_a: numpy array with shape (Nx4x2)
    boxes_b: numpy_array with shape (Mx4x2)
    """
    poly_a = convert_format(center_to_corners(boxes_a))
    poly_b = convert_format(center_to_corners(boxes_b))

    ious = np.zeros((boxes_a.shape[0], boxes_b.shape[0]))

    for i, box_a in enumerate(poly_a):
        for j, box_b in enumerate(poly_b):
            ious[i][j] = box_a.intersection(box_b).area / box_a.union(box_b).area

    return ious




if __name__ == "__main__":
    boxes_a = np.array([[0, 0, 1, 2, 0],
                      [0, 0, 1, 2, math.pi / 2 ],
                      [0, 0, 1, 2, 0]])
    boxes_b = np.array([[0, 0, 1, 2, 0]])
    print(compute_iou(boxes_a, boxes_b))