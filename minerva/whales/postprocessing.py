import numpy as np
from sklearn.externals import joblib

from ..backend.base import BaseTransformer


class Adjuster(BaseTransformer):
    def fit(self, prediction_coordinates, crop_coordinates):
        return self

    def transform(self, prediction_coordinates, crop_coordinates):
        adjusted_coordinates = prediction_coordinates + crop_coordinates
        return {'prediction_coordinates': adjusted_coordinates.astype(np.uint64)}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class UnBinner(BaseTransformer):
    def __init__(self, bins_nr, shape=None):
        self.bins_nr = bins_nr
        self.shape = shape

    def fit(self, prediction_coordinates, original_shapes):
        return self

    def transform(self, prediction_coordinates, original_shapes):
        if self.shape is None:
            y_granularity = original_shapes[['height']].values / self.bins_nr
            x_granularity = original_shapes[['width']].values / self.bins_nr
        else:
            y_granularity = self.shape[1] / self.bins_nr
            x_granularity = self.shape[0] / self.bins_nr

        scaled_coordinates = np.zeros_like(prediction_coordinates)
        scaled_coordinates[:, [0, 2]] = prediction_coordinates[:, [0, 2]] * x_granularity
        scaled_coordinates[:, [1, 3]] = prediction_coordinates[:, [1, 3]] * y_granularity
        return {'prediction_coordinates': scaled_coordinates.astype(np.uint64)}

    def load(self, filepath):
        param_dict = joblib.load(filepath)
        self.bins_nr = param_dict['bins_nr']
        return self

    def save(self, filepath):
        joblib.dump({'bins_nr': self.bins_nr}, filepath)


class DetectionAverage(BaseTransformer):
    def fit(self, predicted_coordinates_list):
        return self

    def transform(self, predicted_coordinates_list):
        average_coordinates = np.mean(predicted_coordinates_list, axis=0).astype(np.int8)
        return {'predicted_coordinates': average_coordinates}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class DetectionRandom(BaseTransformer):
    def fit(self, predicted_coordinates_list):
        return self

    def transform(self, predicted_coordinates_list):
        ensemble_size = predicted_coordinates_list.shape[0]
        random_index = np.random.randint(ensemble_size)
        random_choice_prediction = predicted_coordinates_list[random_index, :, :].astype(np.int8)
        return {'predicted_coordinates': random_choice_prediction}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class AlignerAverage(BaseTransformer):
    def fit(self, predicted_points_list):
        return self

    def transform(self, predicted_points_list):
        average_points = np.mean(predicted_points_list, axis=0).astype(np.int8)
        return {'predicted_points': average_points}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class ProbabilityCalibration(BaseTransformer):
    def __init__(self, power):
        self.power = power

    def fit(self, prediction_proba):
        return self

    def transform(self, prediction_probability):
        prediction_probability = np.array(prediction_probability) ** self.power
        return {'prediction_probability': prediction_probability}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)
