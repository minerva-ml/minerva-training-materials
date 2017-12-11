from functools import partial

import cv2
from imgaug import augmenters as iaa
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from math import sqrt

from .config import GLOBAL_CONFIG

SHAPE_COLUMNS = ['height', 'width']
LOCALIZER_TARGET_COLUMNS = ['bbox1_x', 'bbox1_y', 'bbox2_x', 'bbox2_y']
LOCALIZER_AUXILARY_COLUMNS = []
ALIGNER_TARGET_COLUMNS = ['bonnet_x', 'bonnet_y', 'blowhead_x', 'blowhead_y']
ALIGNER_AUXILARY_COLUMNS = ['callosity', 'whaleID']
CLASSIFIER_TARGET_COLUMNS = ['whaleID']
CLASSIFIER_AUXILARY_COLUMNS = ['callosity']

LOCALIZER_COLUMNS = LOCALIZER_TARGET_COLUMNS + LOCALIZER_AUXILARY_COLUMNS
ALIGNER_COLUMNS = ALIGNER_TARGET_COLUMNS + ALIGNER_AUXILARY_COLUMNS
CLASSIFIER_COLUMNS = CLASSIFIER_TARGET_COLUMNS + CLASSIFIER_AUXILARY_COLUMNS


def add_crop_to_validation(input_):
    """
    Note:
        input is a list by definition
    """
    if input_[0] is not None:
        X, y = input_[0]
        crop_coordinates = X[LOCALIZER_TARGET_COLUMNS].values
        return X, y, crop_coordinates
    else:
        return None


def add_alignment_to_validation(input_):
    """
    Note:
        input is a list by definition
    """
    if input_[0] is not None:
        X, y = input_[0]
        alignment_coordinates = X[ALIGNER_TARGET_COLUMNS].values
        return X, y, alignment_coordinates
    else:
        return None


def get_crop_coordinates(input_):
    if input_[0] is not None:
        return input_[0][LOCALIZER_TARGET_COLUMNS].values
    else:
        return None


def get_align_coordinates(input_):
    if input_[0] is not None:
        return input_[0][ALIGNER_TARGET_COLUMNS].values
    else:
        return None


class CropKeypoints(iaa.Augmenter):
    def __init__(self, crop_keypoints, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.crop_keypoints = tuple([np.int(coord) for coord in crop_keypoints])
        self.crop_keypoints_float = tuple([np.float(coord) for coord in crop_keypoints])

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        for i in range(nb_images):
            kp_left, kp_top, kp_right, kp_bottom = self.crop_keypoints
            image_cr = images[i][kp_top:kp_bottom, kp_left:kp_right, :]
            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            kp_left, kp_top, kp_right, kp_bottom = self.crop_keypoints_float
            shifted = keypoints_on_image.shift(x=-kp_left, y=-kp_top)
            shifted.shape = (kp_bottom - kp_top, kp_right - kp_left) + shifted.shape[2:]
            result.append(shifted)

        return result

    def get_parameters(self):
        return []


class AlignKeypoints(iaa.Augmenter):
    def __init__(self, align_keypoints, target_size, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.align_keypoints = align_keypoints
        self.align_keypoints_float = tuple([np.float(coord) for coord in align_keypoints])
        self.target_size = target_size

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        for i in range(nb_images):
            p1_x, p1_y, p2_x, p2_y = self.align_keypoints
            image_al = align(images[i], (p1_x, p1_y), (p2_x, p2_y),
                             target_size=self.target_size)
            result.append(image_al)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return []

    def get_parameters(self):
        return []


def align(img, aligning_coordinates_p1, aligning_coordinates_p2, target_size=(200, 100)):
    """
    Adjusted from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python
    """
    t_width, t_height = target_size

    target_position = (int(0.75 * t_width), int(0.5 * t_height))

    x1, y1 = aligning_coordinates_p1
    x2, y2 = aligning_coordinates_p2

    left_x, left_y = target_position

    dy = y2 - y1
    dx = x2 - x1
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desired_dist = 0.5 * t_width
    scale = desired_dist / dist

    M = cv2.getRotationMatrix2D((x1, y1), angle, scale)

    M[0, 2] += (left_x - x1)
    M[1, 2] += (left_y - y1)

    output = cv2.warpAffine(img, M, (t_width, t_height), flags=cv2.INTER_CUBIC)

    return output


def rmse_multi(y_pred, y_true):
    rmse = []
    for i in range(y_pred.shape[1]):
        pred = y_pred[:, i]
        true = y_true[:, i]
        rmse_chunk = sqrt(mean_squared_error(pred, true))
        rmse.append(rmse_chunk)
    return np.mean(rmse)


SUB_PROBLEMS_SPECS = {'localization': {'output_name': 'prediction_coordinates',
                                       'target_columns': LOCALIZER_TARGET_COLUMNS,
                                       'score_function': rmse_multi
                                       },
                      'alignment': {'output_name': 'prediction_coordinates',
                                    'target_columns': ALIGNER_TARGET_COLUMNS,
                                    'score_function': rmse_multi
                                    },
                      'classification': {'output_name': 'prediction_probability',
                                         'target_columns': CLASSIFIER_TARGET_COLUMNS,
                                         'score_function': partial(log_loss, labels=GLOBAL_CONFIG['num_classes'])
                                         },
                      'end_to_end': {'output_name': 'prediction_probability',
                                     'target_columns': CLASSIFIER_TARGET_COLUMNS,
                                     'score_function': partial(log_loss, labels=GLOBAL_CONFIG['num_classes'])
                                     },
                      }
