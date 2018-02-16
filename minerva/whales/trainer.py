import json
import os
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from minerva.utils import get_logger
from .config import SHAPE_COLUMNS, LOCALIZER_COLUMNS, ALIGNER_COLUMNS, CLASSIFIER_COLUMNS
from .config import SOLUTION_CONFIG as config
from .validation import SCORE_FUNCTIONS
from ..backend.cross_validation import train_test_split_atleast_one
from ..backend.trainer import BasicTrainer

RANDOM_STATE = 7300
logger = get_logger()


class Trainer(BasicTrainer):
    def __init__(self, pipeline, config, dev_mode=False, cloud_mode=False, sub_problem=None):
        super().__init__(pipeline, config, dev_mode)
        self.cloud_mode = cloud_mode
        self.sub_problem = sub_problem
        self.cv_splitting = partial(train_test_split_atleast_one, test_size=0.12, random_state=RANDOM_STATE)

    def train(self):
        (X_train, y_train), (X_valid, y_valid) = self._load_train_valid()

        self.pipeline.fit_transform({'unbinner_input': {'original_shapes': X_train[SHAPE_COLUMNS],
                                                        },
                                     'localizer_input': {'X': X_train,
                                                         'y': y_train[LOCALIZER_COLUMNS],
                                                         'validation_data': (X_valid,
                                                                             y_valid[LOCALIZER_COLUMNS]),
                                                         'train_mode': True,
                                                         },
                                     'aligner_input': {'X': X_train,
                                                       'y': y_train[ALIGNER_COLUMNS],
                                                       'validation_data': (X_valid,
                                                                           y_valid[ALIGNER_COLUMNS]),
                                                       'train_mode': True,
                                                       },
                                     'classifier_input': {'X': X_train,
                                                          'y': y_train[CLASSIFIER_COLUMNS],
                                                          'validation_data': (X_valid,
                                                                              y_valid[CLASSIFIER_COLUMNS]),
                                                          'train_mode': True,
                                                          }
                                     })

    def _evaluate(self, X, y):
        outputs = self.pipeline.transform({'unbinner_input': {'original_shapes': X[SHAPE_COLUMNS],
                                                              },
                                           'localizer_input': {'X': X,
                                                               'y': y[LOCALIZER_COLUMNS],
                                                               'validation_data': None,
                                                               'train_mode': False,
                                                               },
                                           'aligner_input': {'X': X,
                                                             'y': y[ALIGNER_COLUMNS],
                                                             'validation_data': None,
                                                             'train_mode': False,
                                                             },
                                           'classifier_input': {'X': X,
                                                                'y': y[CLASSIFIER_COLUMNS],
                                                                'validation_data': None,
                                                                'train_mode': False,
                                                                }
                                           })
        y_pred = outputs['y_pred']
        y_true = outputs['y_true']
        score = SCORE_FUNCTIONS[self.sub_problem](y_true, y_pred)
        return score

    def _load_train_valid(self):
        (X_train, y_train), _ = load_whale_data(self.cloud_mode)
        X_train_, X_valid_, y_train_, y_valid_ = self.cv_splitting(X_train, y_train)
        return (X_train_, y_train_), (X_valid_, y_valid_)

    def _load_test(self):
        _, (X_test, y_test) = load_whale_data(self.cloud_mode)
        return X_test, y_test

    def _load_grid_search_params(self):
        n_iter = self.config.GRID_SEARCH_CONFIG['n_iter']
        grid_params = self.config.GRID_SEARCH_CONFIG['params']
        return n_iter, grid_params


def load_whale_data(cloud_mode):
    meta_filepath = config['trainer']['metadata']

    if cloud_mode:
        meta_filepath = meta_filepath.replace('metadata', 'meta_data')

    meta_data_ = pd.read_csv(meta_filepath)
    meta_data = meta_data_.reset_index(drop=True)

    X = meta_data
    y = meta_data[['bbox1_x', 'bbox1_y', 'bbox2_x', 'bbox2_y',
                   'bonnet_x', 'bonnet_y', 'blowhead_x', 'blowhead_y',
                   'whaleID', 'callosity']]
    X_train, X_test, y_train, y_test = train_test_split_atleast_one(X, y, test_size=0.1, random_state=RANDOM_STATE)
    return (X_train, y_train), (X_test, y_test)


def generate_metadata():
    def _generate_bboxes():
        df_bbox = pd.DataFrame(columns=['bbox1_x', 'bbox1_y', 'bbox2_x', 'bbox2_y'])
        df_bbox.index.name = 'Image'
        with open(config['trainer']['bbox_train_json'], 'r') as data_file:
            data = json.load(data_file)
        for j in range(len(data)):
            img_name = data[j]['filename']
            bbox1_x = data[j]['annotations'][0]['x']
            bbox1_y = data[j]['annotations'][0]['y']
            bbox2_x = data[j]['annotations'][0]['x'] + data[j]['annotations'][0]['width']
            bbox2_y = data[j]['annotations'][0]['y'] + data[j]['annotations'][0]['height']
            df_bbox.loc[img_name] = [bbox1_x, bbox1_y, bbox2_x, bbox2_y]
        return df_bbox

    def _generate_train_test_split():
        df_data_split = pd.read_csv(config['trainer']['train_csv'], dtype=str, index_col='Image')
        df_data_split.drop(labels='whaleID', axis=1, inplace=True)
        img_names = df_data_split.index.values
        train_imgs, test_imgs = train_test_split_atleast_one(img_names, test_size=0.1,
                                                             random_state=RANDOM_STATE)

        df_data_split.insert(len(df_data_split.columns), 'is_train', 0)
        df_data_split.insert(len(df_data_split.columns), 'is_test', 0)

        df_data_split.loc[train_imgs.tolist(), 'is_train'] = 1
        df_data_split.loc[test_imgs.tolist(), 'is_test'] = 1
        return df_data_split

    def _generate_points():
        df_points = pd.DataFrame(columns=['bonnet_x', 'bonnet_y', 'blowhead_x', 'blowhead_y'])
        df_points.index.name = 'Image'
        with open(config['trainer']['bonnet_tip_json'], 'r') as data_file1:
            data_bonnet = json.load(data_file1)
        for j in range(len(data_bonnet)):
            img_name = data_bonnet[j]['filename']
            bonnet_x = data_bonnet[j]['annotations'][0]['x']
            bonnet_y = data_bonnet[j]['annotations'][0]['y']
            df_points.loc[img_name, ['bonnet_x', 'bonnet_y']] = {'bonnet_x': bonnet_x,
                                                                 'bonnet_y': bonnet_y}

        with open(config['trainer']['blowhead_json'], 'r') as data_file2:
            data_blowhead = json.load(data_file2)
        for j in range(len(data_blowhead)):
            img_name = data_blowhead[j]['filename']
            blowhead_x = data_blowhead[j]['annotations'][0]['x']
            blowhead_y = data_blowhead[j]['annotations'][0]['y']
            df_points.loc[img_name, ['blowhead_x', 'blowhead_y']] = {'blowhead_x': blowhead_x,
                                                                     'blowhead_y': blowhead_y}
        return df_points

    def _generate_callosity():
        df_callosity = pd.read_csv(config['trainer']['callosity_csv'], index_col='name')
        df_callosity.columns = ['callosity']
        return df_callosity

    def _generate_original_shape():
        meta = pd.read_csv(config['trainer']['train_csv'], dtype=str, index_col='Image')
        meta.drop('w_7489.jpg', inplace=True)
        meta.reset_index(inplace=True)
        height, width = [], []
        for i, image_name in enumerate(meta['Image'].values):
            logger.info('{}/{}'.format(i, meta.shape[0]))
            img_filepath = os.path.join(config['trainer']['imgs_dir'], image_name)
        meta = pd.read_csv(config['trainer']['train_csv'], dtype=str, index_col='Image')

        height, width = [], []
        for image_name in tqdm(meta['Image'].values):
            img_filepath = os.path.join(config['trainer']['imgs_dir'], image_name)
            img = plt.imread(img_filepath)
            h, w = img.shape[:2]
            height.append(h)
            width.append(w)
        meta['height'] = height
        meta['width'] = width

        meta.set_index('Image', inplace=True)
        return meta

    df_metadata = pd.read_csv(config['trainer']['train_csv'], dtype=str, index_col='Image')
    df_metadata.drop('w_7489.jpg', inplace=True)  # missing image

    logger.info('Adding bboxes')
    df_metadata = df_metadata.join(_generate_bboxes())
    logger.info('Adding other annotations')
    df_metadata = df_metadata.join(_generate_points())
    logger.info('Adding callosity')
    df_metadata = df_metadata.join(_generate_callosity(), sort=True)
    # logger.info('Adding original shape')
    # df_metadata = df_metadata.join(_generate_original_shape(), sort=True) # Todo not working set indexx or smth
    df_metadata.reset_index(inplace=True)
    df_metadata = df_metadata.join(_generate_callosity(), sort=True)
    df_metadata = df_metadata[['whaleID', 'is_train',
                               'height', 'width',
                               'callosity',
                               'bbox1_x', 'bbox1_y', 'bbox2_x', 'bbox2_y',
                               'bonnet_x', 'bonnet_y',
                               'blowhead_x', 'blowhead_y']]
    # logger.info('Train test split')
    # df_metadata['is_train'] = _generate_train_test_split(df_metadata['whaleID'].values)
    return df_metadata
