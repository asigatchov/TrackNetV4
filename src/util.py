"""
Utility functions for dataset generation, training utilities, and model/dataset management.
"""

import math
import numpy as np
import cv2
import tensorflow.keras.backend as K

from constants import BADMINTON_DATASET_ROOT, TENNIS_DATASET_ROOT, NEW_TENNIS_DATASET_ROOT, WIDTH, HEIGHT
from models.TrackNetV2 import TrackNetV2
from models.TrackNetV4 import TrackNetV4

####################################
# Dataset related helper functions #
####################################

def genHeatMap(w, h, cx, cy, r, mag):
    """
    Generate a heatmap with a circular region set to a specified magnitude.
    If the center coordinates (cx, cy) are negative, the function returns a zero-filled heatmap.
    """
    if cx < 0 or cy < 0:
        return np.zeros((h, w))

    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap * mag

##############################
# Training related functions #
##############################

def outcome(y_pred, y_true, tol):
    """
    Calculate the outcomes (TP, TN, FP1, FP2, FN) for a batch of predicted heatmaps versus ground truth.
    """
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
                TN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
                FP2 += 1
            elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
                FN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
                h_pred = y_pred[i][j] * 255
                h_true = y_true[i][j] * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)

def custom_loss(y_true, y_pred):
    """
    Custom loss function for TrackNet training.
    """
    loss = (-1) * (K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) +
                  K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return loss

###################################
# Model/dataset related functions #
###################################

def get_model(model_name, height, width):
    """
    Retrieve an instance of a TrackNet model based on the specified model name.
    """
    if model_name == "Baseline_TrackNetV2":
        return TrackNetV2(height, width)
    elif model_name == "TrackNetV4_TypeA":
        return TrackNetV4(height, width, "TypeA")
    elif model_name == "TrackNetV4_TypeB":
        return TrackNetV4(height, width, "TypeB")
    elif model_name == "TrackNetV4_Nano":
        from models.TrackNetV4Nano import TrackNetV4Nano
        return TrackNetV4Nano(height, width, fusion_layer_type="TypeA")
    elif model_name == "TrackNetV4_Small":
        from models.TrackNetV4Small import TrackNetV4Small
        return TrackNetV4Small(height, width, fusion_layer_type="TypeA")
   
    else:
        raise ValueError("Unknown model name")

def get_dataset(dataset_name, mode, height=HEIGHT, width=WIDTH):
    """
    Retrieve an instance of a dataset based on the provided dataset name and mode.
    """
    from dataset import TennisDataset, BadmintonDataset, NewTennisDataset

    if dataset_name == "tennis_game_level_split":
        return TennisDataset(TENNIS_DATASET_ROOT, "game_level", mode,
                             target_img_height=height, target_img_width=width)
    elif dataset_name == "tennis_clip_level_split":
        return TennisDataset(TENNIS_DATASET_ROOT, "clip_level", mode,
                             target_img_height=height, target_img_width=width)
    elif dataset_name == "new_tennis":
        return NewTennisDataset(NEW_TENNIS_DATASET_ROOT, mode,
                                target_img_height=height, target_img_width=width)
    elif dataset_name == "badminton":
        return BadmintonDataset(BADMINTON_DATASET_ROOT, mode,
                                target_img_height=height, target_img_width=width)
    else:
        raise ValueError("Unknown dataset name")
