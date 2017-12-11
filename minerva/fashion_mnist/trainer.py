import gzip
import os
from functools import partial

import numpy as np
from keras.utils.data_utils import get_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..backend.trainer import BasicTrainer


class Trainer(BasicTrainer):
    def __init__(self, pipeline, config, dev_mode=False):
        super().__init__(pipeline, config, dev_mode)
        self.cv_splitting = partial(train_test_split, test_size=0.2, random_state=1234)

    def train(self):
        (X_train, y_train), (X_valid, y_valid) = self._load_train_valid()
        self.pipeline.fit_transform({'data': {'X': X_train,
                                              'y': y_train,
                                              'validation_data': (X_valid, y_valid),
                                              'inference': False}})

    def _evaluate(self, X, y):
        predictions = self.pipeline.transform({'data': {'X': X,
                                                        'y': None,
                                                        'validation_data': None,
                                                        'inference': True}})
        y_pred = predictions['y_pred']
        score = accuracy_score(y_pred, y)
        return score

    def _load_train_valid(self):
        (X_train, y_train), _ = load_data()
        if self.dev_mode:
            X_train = X_train[:1024]
            y_train = y_train[:1024]
        X_train_, X_valid_, y_train_, y_valid_ = self.cv_splitting(X_train, y_train)
        return (X_train_, y_train_), (X_valid_, y_valid_)

    def _load_test(self):
        _, (X_test, y_test) = load_data()
        return X_test, y_test

    def _load_grid_search_params(self):
        n_iter = self.config.GRID_SEARCH_CONFIG['n_iter']
        grid_params = self.config.GRID_SEARCH_CONFIG['params']
        return n_iter, grid_params


def load_data():
    """Loads the Fashion-MNIST dataset.
    returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for file in files:
        paths.append(get_file(file, origin=base + file, cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
