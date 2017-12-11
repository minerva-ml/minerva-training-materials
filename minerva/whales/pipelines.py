from .models import SimpleLocalizer, SimpleAligner, SimpleClassifier

from .postprocessing import DetectionAverage, AlignerAverage, ProbabilityCalibration, UnBinner, Adjuster
from .preprocessing import TargetEncoderPandas, DataLoaderLocalizer, DataLoaderAligner, DataLoaderClassifier
from .utils import get_crop_coordinates, add_crop_to_validation, get_align_coordinates, add_alignment_to_validation
from ..backend.base import SubstitutableStep, stack_inputs, identity_inputs
from ..backend.postprocessing import PredictionAverage


def localization_pipeline(config):
    dataloader = SubstitutableStep(name='localizer_loader',
                                   transformer=DataLoaderLocalizer(**config['localizer_dataloader']),
                                   input_data=['localizer_input'],
                                   cache_dirpath=config['global']['cache_dirpath'])
    network = SubstitutableStep(name='localizer_network',
                                transformer=SimpleLocalizer(**config['localizer_network']),
                                input_steps=[dataloader],
                                cache_dirpath=config['global']['cache_dirpath'])
    unbinner = SubstitutableStep(name='localizer_unbinner',
                                 transformer=UnBinner(**config['localizer_unbinner']),
                                 input_steps=[network],
                                 input_data=['unbinner_input'],
                                 cache_dirpath=config['global']['cache_dirpath'])
    return unbinner


def alignment_pipeline(config):
    encoder = SubstitutableStep(name='aligner_encoder',
                                transformer=TargetEncoderPandas(**config['aligner_encoder']),
                                input_data=['aligner_input'],
                                adapter={'X': ([('aligner_input', 'X')], identity_inputs),
                                         'y': ([('aligner_input', 'y')], identity_inputs),
                                         'validation_data': ([('aligner_input', 'validation_data')], identity_inputs),
                                         },
                                cache_dirpath=config['global']['cache_dirpath'])
    dataloader = SubstitutableStep(name='aligner_loader',
                                   transformer=DataLoaderAligner(**config['aligner_dataloader']),
                                   input_steps=[encoder],
                                   input_data=['aligner_input'],
                                   adapter={'X': ([('aligner_encoder', 'X')], identity_inputs),
                                            'y': ([('aligner_encoder', 'y')], identity_inputs),
                                            'crop_coordinates': ([('aligner_encoder', 'X')],
                                                                 get_crop_coordinates),
                                            'validation_data': ([('aligner_encoder', 'validation_data')],
                                                                add_crop_to_validation),
                                            'train_mode': ([('aligner_input', 'train_mode')], identity_inputs)
                                            },
                                   cache_dirpath=config['global']['cache_dirpath'])
    network = SubstitutableStep(name='aligner_network',
                                transformer=SimpleAligner(**config['aligner_network']),
                                input_steps=[dataloader],
                                cache_dirpath=config['global']['cache_dirpath'])
    unbinner = SubstitutableStep(name='aligner_unbinner',
                                 transformer=UnBinner(**config['aligner_unbinner']),
                                 input_steps=[network],
                                 input_data=['unbinner_input'],
                                 cache_dirpath=config['global']['cache_dirpath'])
    adjuster = SubstitutableStep(name='aligner_adjuster',
                                 transformer=Adjuster(),
                                 input_steps=[unbinner],
                                 input_data=['aligner_input'],
                                 adapter={'crop_coordinates': ([('aligner_input', 'X')],
                                                               get_crop_coordinates),
                                          'prediction_coordinates': (
                                              [('aligner_unbinner', 'prediction_coordinates')], identity_inputs)
                                          },
                                 cache_dirpath=config['global']['cache_dirpath'])
    return adjuster


def classification_pipeline(config):
    encoder = SubstitutableStep(name='classifier_encoder',
                                transformer=TargetEncoderPandas(**config['classifier_encoder']),
                                input_data=['classifier_input'],
                                adapter={'X': ([('classifier_input', 'X')], identity_inputs),
                                         'y': ([('classifier_input', 'y')], identity_inputs),
                                         'validation_data': (
                                             [('classifier_input', 'validation_data')], identity_inputs)
                                         },
                                cache_dirpath=config['global']['cache_dirpath'])
    dataloader = SubstitutableStep(name='classifier_loader',
                                   transformer=DataLoaderClassifier(**config['classifier_dataloader']),
                                   input_steps=[encoder],
                                   input_data=['classifier_input'],
                                   adapter={'X': ([('classifier_encoder', 'X')], identity_inputs),
                                            'y': ([('classifier_encoder', 'y')], identity_inputs),
                                            'align_coordinates': ([('classifier_encoder', 'X')], get_align_coordinates),
                                            'validation_data': ([('classifier_encoder', 'validation_data')],
                                                                add_alignment_to_validation),
                                            'train_mode': ([('classifier_input', 'train_mode')], identity_inputs),
                                            },
                                   cache_dirpath=config['global']['cache_dirpath'])
    network = SubstitutableStep(name='classifier_network',
                                transformer=SimpleClassifier(**config['classifier_network']),
                                input_steps=[dataloader],
                                cache_dirpath=config['global']['cache_dirpath'])
    proba_calibrator = SubstitutableStep(name='classifier_calibrator',
                                         transformer=ProbabilityCalibration(**config['probability_calibration']),
                                         input_steps=[network],
                                         cache_dirpath=config['global']['cache_dirpath'])
    return proba_calibrator


def end_to_end_pipeline(config):
    localizer_loader = SubstitutableStep(name='localizer_loader',
                                         transformer=DataLoaderLocalizer(**config['localizer_dataloader']),
                                         input_data=['localizer_input'],
                                         cache_dirpath=config['global']['cache_dirpath'])
    localizer_network = SubstitutableStep(name='localizer_network',
                                          transformer=SimpleLocalizer(**config['localizer_network']),
                                          input_steps=[localizer_loader],
                                          cache_dirpath=config['global']['cache_dirpath'])
    localizer_unbinner = SubstitutableStep(name='localizer_unbinner',
                                           transformer=UnBinner(**config['localizer_unbinner']),
                                           input_steps=[localizer_network],
                                           input_data=['unbinner_input'],
                                           cache_dirpath=config['global']['cache_dirpath'])

    aligner_encoder = SubstitutableStep(name='aligner_encoder',
                                        transformer=TargetEncoderPandas(**config['aligner_encoder']),
                                        input_steps=['aligner_input'],
                                        adapter={'X': ([('aligner_input', 'X')], identity_inputs),
                                                 'y': ([('aligner_input', 'y')], identity_inputs),
                                                 'validation_data': (
                                                     [('aligner_input', 'validation_data')], identity_inputs),
                                                 },
                                        cache_dirpath=config['global']['cache_dirpath'])
    aligner_loader = SubstitutableStep(name='aligner_loader',
                                       transformer=DataLoaderAligner(**config['aligner_dataloader']),
                                       input_steps=[aligner_encoder, localizer_unbinner],
                                       input_data=['aligner_input'],
                                       adapter={'X': ([('aligner_encoder', 'X')], identity_inputs),
                                                'y': ([('aligner_encoder', 'y')], identity_inputs),
                                                'crop_coordinates': ([('localizer_unbinner', 'X')],
                                                                     get_crop_coordinates),
                                                'validation_data': ([('aligner_encoder', 'validation_data')],
                                                                    add_crop_to_validation),
                                                'train_mode': ([('aligner_input', 'train_mode')], identity_inputs)
                                                },
                                       cache_dirpath=config['global']['cache_dirpath'])
    aligner_network = SubstitutableStep(name='aligner_network',
                                        transformer=SimpleAligner(**config['aligner_network']),
                                        input_steps=[aligner_loader],
                                        cache_dirpath=config['global']['cache_dirpath'])
    aligner_unbinner = SubstitutableStep(name='aligner_unbinner',
                                         transformer=UnBinner(**config['aligner_unbinner']),
                                         input_steps=[aligner_network],
                                         input_data=['unbinner_input'],
                                         cache_dirpath=config['global']['cache_dirpath'])
    aligner_adjuster = SubstitutableStep(name='aligner_adjuster',
                                         transformer=Adjuster(),
                                         input_steps=[aligner_unbinner],
                                         input_data=['aligner_input'],
                                         adapter={'crop_coordinates': ([('aligner_input', 'X')],
                                                                       get_crop_coordinates)},
                                         cache_dirpath=config['global']['cache_dirpath'])

    classifier_encoder = SubstitutableStep(name='classifier_encoder',
                                           transformer=TargetEncoderPandas(**config['classifier_encoder']),
                                           input_data=['classifier_input'],
                                           adapter={'X': ([('classifier_input', 'X')], identity_inputs),
                                                    'y': ([('classifier_input', 'y')], identity_inputs),
                                                    'validation_data': (
                                                        [('classifier_input', 'validation_data')], identity_inputs)
                                                    },
                                           cache_dirpath=config['global']['cache_dirpath'])
    classifier_loader = SubstitutableStep(name='classifier_loader',
                                          transformer=DataLoaderClassifier(**config['classifier_dataloader']),
                                          input_steps=[classifier_encoder],
                                          input_data=['classifier_input'],
                                          adapter={'X': ([('classifier_encoder', 'X')], identity_inputs),
                                                   'y': ([('classifier_encoder', 'y')], identity_inputs),
                                                   'align_coordinates': (
                                                       [('aligner_adjuster', 'X')], get_align_coordinates),
                                                   'validation_data': ([('classifier_encoder', 'validation_data')],
                                                                       add_alignment_to_validation),
                                                   'train_mode': (
                                                       [('classifier_input', 'train_mode')], identity_inputs),
                                                   },
                                          cache_dirpath=config['global']['cache_dirpath'])
    classifier_network = SubstitutableStep(name='classifier_network',
                                           transformer=SimpleClassifier(**config['classifier_network']),
                                           input_steps=[classifier_loader],
                                           cache_dirpath=config['global']['cache_dirpath'])
    proba_calibrator = SubstitutableStep(name='proba_calibrator',
                                         transformer=ProbabilityCalibration(**config['probability_calibration']),
                                         input_steps=[classifier_network],
                                         cache_dirpath=config['global']['cache_dirpath'])

    return proba_calibrator
