import os
import yaml
from datetime import datetime

with open('neptune.yaml') as f:
    config = yaml.load(f)
exp_name = config['name']
exp_root = config['parameters']['solution_dir']
data_dir = config['parameters']['data_dir']
os.makedirs(exp_root, exist_ok=True)

# exp_name = 'exp_early'
#
# exp_start = datetime.now().strftime("%Y%m%d-%H%M%S")
# exp_root = '/mnt/ml-team/minerva/cache/whales/experiments/{}'.format(exp_start + '_' + exp_name)
# os.makedirs(exp_root, exist_ok=True)

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

TARGET_COLUMNS = {'localization': LOCALIZER_TARGET_COLUMNS,
                  'alignment': ALIGNER_TARGET_COLUMNS,
                  'classification': CLASSIFIER_TARGET_COLUMNS,
                  'end_to_end': CLASSIFIER_TARGET_COLUMNS
                  }

GLOBAL_CONFIG = {'exp_name': exp_name,
                 'exp_root': exp_root,
                 'num_workers': 6,
                 'callosity_classes': 3,
                 'num_classes': 447,
                 'img_H-W': (256, 256),
                 'img_C-H-W': (3, 256, 256),
                 'batch_size_train': 32,
                 'batch_size_inference': 32,
                 'localizer_bins': 128,
                 'aligner_bins': 128
                 }

SOLUTION_CONFIG = {
    'global': {'cache_dirpath': GLOBAL_CONFIG['exp_root']},
    'trainer': {'metadata': os.path.join(data_dir, 'metadata.csv'),
                'train_csv': os.path.join(data_dir, 'annotations/train.csv'),
                'bbox_train_json': os.path.join(data_dir, 'annotations/slot.json'),
                'bonnet_tip_json': os.path.join(data_dir, 'annotations/bonnet_tip.json'),
                'blowhead_json': os.path.join(data_dir, 'annotations/blowhead.json'),
                'callosity_csv': os.path.join(data_dir, 'annotations/new_conn.csv'),
                'imgs_dir': os.path.join(data_dir, 'imgs')
                },
    'localizer_dataloader': {'dataset_params': {'train': {'img_dirpath': os.path.join(data_dir, 'imgs'),
                                                          'augmentation': True,
                                                          'target_size': GLOBAL_CONFIG['img_H-W'],
                                                          'bins_nr': GLOBAL_CONFIG['localizer_bins']
                                                          },
                                                'inference': {'img_dirpath': os.path.join(data_dir, 'imgs'),
                                                              'augmentation': False,
                                                              'target_size': GLOBAL_CONFIG['img_H-W'],
                                                              'bins_nr': GLOBAL_CONFIG['localizer_bins']
                                                              },
                                                },
                             'loader_params': {'train': {'batch_size': GLOBAL_CONFIG['batch_size_train'],
                                                         'shuffle': True,
                                                         'num_workers': GLOBAL_CONFIG['num_workers']
                                                         },
                                               'inference': {'batch_size': GLOBAL_CONFIG['batch_size_inference'],
                                                             'shuffle': False,
                                                             'num_workers': GLOBAL_CONFIG['num_workers']
                                                             },
                                               },
                             },
    'localizer_network': {'architecture_config': {'model_params': {'input_shape': GLOBAL_CONFIG['img_C-H-W'],
                                                                   'classes': GLOBAL_CONFIG['localizer_bins']
                                                                   },
                                                  'optimizer_params': {'lr': 0.0005,
                                                                       'momentum': 0.9,
                                                                       'nesterov': True
                                                                       },
                                                  'regularizer_params': {'regularize': True,
                                                                         'weight_decay_conv2d': 0.0005,
                                                                         'weight_decay_linear': 0.01},
                                                  'weights_init': {'function': 'normal',
                                                                   'params': {'mean': 0,
                                                                              'std_conv2d': 0.01,
                                                                              'std_linear': 0.001
                                                                              },
                                                                   },
                                                  },
                          'training_config': {'epochs': 150},
                          'callbacks_config': {'model_checkpoint': {
                              'checkpoint_dir': os.path.join(exp_root, 'checkpoints', 'localizer_network'),
                              'epoch_every': 1},
                              'lr_scheduler': {'gamma': 0.9955,
                                               'epoch_every': 1},
                              'training_monitor': {'batch_every': 1,
                                                   'epoch_every': 1},
                              'validation_monitor': {'epoch_every': 1},
                              'bounding_box_predictions': {'img_dir': 'output/debugging',
                                                           'bins_nr': GLOBAL_CONFIG['localizer_bins'],
                                                           'epoch_every': 1
                                                           },
                              'neptune_monitor': {'bins_nr': GLOBAL_CONFIG['localizer_bins'],
                                                  'img_nr': 10}
                          },
                          },
    'localizer_unbinner': {'bins_nr': GLOBAL_CONFIG['localizer_bins']},

    'aligner_encoder': {'encode': ['callosity', 'whaleID'],
                        'no_encode': ['bonnet_x', 'bonnet_y', 'blowhead_x', 'blowhead_y', ],
                        },
    'aligner_dataloader': {'dataset_params': {'train': {'img_dirpath': os.path.join(data_dir, 'imgs'),
                                                        'augmentation': True,
                                                        'target_size': GLOBAL_CONFIG['img_H-W'],
                                                        'bins_nr': GLOBAL_CONFIG['aligner_bins']
                                                        },
                                              'inference': {'img_dirpath': os.path.join(data_dir, 'imgs'),
                                                            'augmentation': False,
                                                            'target_size': GLOBAL_CONFIG['img_H-W'],
                                                            'bins_nr': GLOBAL_CONFIG['aligner_bins']
                                                            },
                                              },
                           'loader_params': {'train': {'batch_size': GLOBAL_CONFIG['batch_size_train'],
                                                       'shuffle': True,
                                                       'num_workers': GLOBAL_CONFIG['num_workers']
                                                       },
                                             'inference': {'batch_size': GLOBAL_CONFIG['batch_size_inference'],
                                                           'shuffle': False,
                                                           'num_workers': GLOBAL_CONFIG['num_workers']
                                                           },
                                             },
                           },
    'aligner_network': {'architecture_config': {'model_params': {'input_shape': GLOBAL_CONFIG['img_C-H-W'],
                                                                 'classes': {'points': GLOBAL_CONFIG['aligner_bins'],
                                                                             'callosity': GLOBAL_CONFIG[
                                                                                 'callosity_classes'],
                                                                             'whale_id': GLOBAL_CONFIG['num_classes']
                                                                             }
                                                                 },
                                                'optimizer_params': {'lr': 0.0005,
                                                                     'momentum': 0.9,
                                                                     'nesterov': True
                                                                     },
                                                'regularizer_params': {'regularize': True,
                                                                       'weight_decay_conv2d': 0.0005,
                                                                       'weight_decay_linear': 0.01},
                                                'weights_init': {'function': 'normal',
                                                                 'params': {'mean': 0,
                                                                            'std_conv2d': 0.01,
                                                                            'std_linear': 0.001
                                                                            },
                                                                 },
                                                },
                        'training_config': {'epochs': 1000},
                        'callbacks_config': {'model_checkpoint': {
                            'checkpoint_dir': os.path.join(exp_root, 'checkpoints', 'aligner_network'),
                            'epoch_every': 1
                        },
                            'lr_scheduler': {'gamma': 0.9955,
                                             'epoch_every': 1},
                            'training_monitor': {'batch_every': 1,
                                                 'epoch_every': 1},
                            'validation_monitor': {'epoch_every': 1},
                            'neptune_monitor': {'bins_nr': GLOBAL_CONFIG['aligner_bins'],
                                                'img_nr': 10}
                        },
                        },
    'aligner_unbinner': {'bins_nr': GLOBAL_CONFIG['aligner_bins'],
                         'shape': GLOBAL_CONFIG['img_H-W']},

    'classifier_encoder': {'encode': ['whaleID', 'callosity'],
                           'no_encode': [],
                           },
    'classifier_dataloader': {'dataset_params': {'train': {'img_dirpath': os.path.join(data_dir, 'imgs'),
                                                           'augmentation': True,
                                                           'target_size': GLOBAL_CONFIG['img_H-W'],
                                                           'num_classes': GLOBAL_CONFIG['num_classes']
                                                           },
                                                 'inference': {'img_dirpath': os.path.join(data_dir, 'imgs'),
                                                               'augmentation': False,
                                                               'target_size': GLOBAL_CONFIG['img_H-W'],
                                                               'num_classes': GLOBAL_CONFIG['num_classes']
                                                               },
                                                 },
                              'loader_params': {'train': {'batch_size': GLOBAL_CONFIG['batch_size_train'],
                                                          'shuffle': True,
                                                          'num_workers': GLOBAL_CONFIG['num_workers']
                                                          },
                                                'inference': {'batch_size': GLOBAL_CONFIG['batch_size_inference'],
                                                              'shuffle': False,
                                                              'num_workers': GLOBAL_CONFIG['num_workers']
                                                              },
                                                },
                              },
    'classifier_network': {'architecture_config': {'model_params': {'input_shape': GLOBAL_CONFIG['img_C-H-W'],
                                                                    'classes': {
                                                                        'whale_id': GLOBAL_CONFIG['num_classes'],
                                                                        'callosity': GLOBAL_CONFIG['callosity_classes'],
                                                                    },
                                                                    },
                                                   'weights_init': {'function': 'normal',
                                                                    'params': {'mean': 0.0,
                                                                               'std_conv2d': 0.01,
                                                                               'std_linear': 0.001
                                                                               },
                                                                    },
                                                   'regularizer_params': {'regularize': True,
                                                                          'weight_decay_conv2d': 0.0005,
                                                                          'weight_decay_linear': 0.01
                                                                          },
                                                   'optimizer_params': {'lr': 0.001,
                                                                        'momentum': 0.9,
                                                                        'nesterov': True
                                                                        },
                                                   },
                           'training_config': {'epochs': 250},
                           'callbacks_config': {'model_checkpoint': {
                               'checkpoint_dir': os.path.join(exp_root, 'checkpoints', 'classifier_network'),
                               'epoch_every': 5,
                               'batch_every': 0
                           },
                               'lr_scheduler': {'gamma': 0.9955},
                               'validation_monitor': {'epoch_every': 1,
                                                      'batch_every': 30
                                                      },
                               'training_monitor': {'epoch_every': 1,
                                                    'batch_every': 30
                                                    },
                           },
                           },
    'classifier_calibrator': {'power': 1.35},
}
