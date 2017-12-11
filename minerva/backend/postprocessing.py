import numpy as np
from sklearn.externals import joblib

from .base import BaseTransformer


class ClassPredictor(BaseTransformer):
    def transform(self, prediction_proba):
        predictions_class = np.argmax(prediction_proba, axis=1)
        return {'y_pred': predictions_class}

    def load(self, filepath):
        return ClassPredictor()

    def save(self, filepath):
        joblib.dump({}, filepath)


class PredictionAverage(BaseTransformer):
    def transform(self, prediction_proba_list):
        avg_pred = np.mean(prediction_proba_list, axis=0)
        return {'prediction_proba': avg_pred}

    def load(self, filepath):
        return PredictionAverage()

    def save(self, filepath):
        joblib.dump({}, filepath)
