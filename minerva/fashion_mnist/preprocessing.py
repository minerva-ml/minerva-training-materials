from math import ceil

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib

from ..backend.base import BaseTransformer


class KerasDataLoader(BaseTransformer):
    def __init__(self, num_classes,
                 target_size,
                 augmentation):
        self.num_classes = num_classes
        self.target_size = target_size
        self.augmentation = augmentation
        self.datagen_builder = build_single_datagen

    def fit(self, X, y, validation_data, inference=False):
        return self

    def transform(self, X, y, validation_data=None, inference=False):
        inference_datagen_args = self.augmentation['inference']['datagen']
        inference_flow_args = self.augmentation['inference']['flow']
        datagen_args = self.augmentation['train']['datagen']
        flow_args = self.augmentation['train']['flow']

        if y is None:
            y = np.zeros((X.shape[0], 1))
            y = self._prep_targets(y)
            X = self._prep_images(X)
            flow, steps = self.datagen_builder(X, y, inference_datagen_args, inference_flow_args)
        else:
            y = self._prep_targets(y)
            X = self._prep_images(X)
            flow, steps = self.datagen_builder(X, y, datagen_args, flow_args)

        if validation_data is not None:
            X_valid, y_valid = validation_data
            y_valid = self._prep_targets(y_valid)
            X_valid = self._prep_images(X_valid)
            valid_flow, valid_steps = self.datagen_builder(X_valid, y_valid,
                                                           inference_datagen_args, inference_flow_args)
        else:
            valid_flow = None
            valid_steps = None

        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def _prep_targets(self, y):
        targets = to_categorical(np.array(y), num_classes=self.num_classes)
        return targets

    def _prep_images(self, X):
        tensor_images = np.expand_dims(X, axis=4)
        return tensor_images

    def load(self, filepath):
        params = joblib.load(filepath)
        return KerasDataLoader(**params)

    def save(self, filepath):
        params = {'num_classes': self.num_classes,
                  'target_size': self.target_size,
                  'augmentation': self.augmentation}
        joblib.dump(params, filepath)


def build_single_datagen(X, y, datagen_args, flow_args):
    datagen = ImageDataGenerator(**datagen_args)

    flow = datagen.flow(X, y, **flow_args)
    steps = ceil(X.shape[0] / flow_args['batch_size'])

    return flow, steps
