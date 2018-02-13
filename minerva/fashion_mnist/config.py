import yaml

with open('neptune.yaml') as f:
    config = yaml.load(f)
exp_name = config['name']
exp_root = config['parameters']['solution_dir']
data_dir = config['parameters']['data_dir']

GLOBAL_CONFIG = {'exp_name': exp_name,
                 'exp_root': exp_root,
                 'num_classes': 10,
                 'input_size': 28,
                 'batch_size': 256,
                 }

SOLUTION_CONFIG = {
    'global': {'cache_dirpath': GLOBAL_CONFIG['exp_root']},
    'loader': {'num_classes': GLOBAL_CONFIG['num_classes'],
               'target_size': GLOBAL_CONFIG['input_size'],
               'augmentation': {'train': {'datagen': {'rescale': 1. / 255,
                                                      'rotation_range': 10,
                                                      'width_shift_range': 0.2,
                                                      'height_shift_range': 0.2,
                                                      },
                                          'flow': {'shuffle': True,
                                                   'batch_size': GLOBAL_CONFIG['batch_size'],
                                                   },
                                          },
                                'inference': {'datagen': {'rescale': 1. / 255
                                                          },
                                              'flow': {'shuffle': False,
                                                       'batch_size': GLOBAL_CONFIG['batch_size'],
                                                       },
                                              },
                                },
               },
    'model': {'architecture_config': {'model_params': {'classes': GLOBAL_CONFIG['num_classes'],
                                                       'input_size': GLOBAL_CONFIG['input_size']
                                                       },
                                      'optimizer_params': {'lr': 0.01,
                                                           'momentum': 0.9,
                                                           'nesterov': True
                                                           },
                                      },
              'training_config': {'epochs': 200},
              'callbacks_config': {'patience': 10,
                                   'model_name': 'simplenet'}
              },
}
