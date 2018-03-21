from .models import SimpleLocalizer, SimpleAligner, SimpleClassifier
from .postprocessing import ProbabilityCalibration, UnBinner, Adjuster
from .preprocessing import TargetEncoderPandas, DataLoaderLocalizer, DataLoaderAligner, DataLoaderClassifier
from .utils import get_crop_coordinates, add_crop_to_validation, get_align_coordinates, add_alignment_to_validation, \
    get_localizer_target_column, get_aligner_target_column, get_classifier_target_column
from ..backend.base import SubstitutableStep, Dummy, identity_inputs, exp_transform


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

    output = SubstitutableStep(name='localizer_output',
                               transformer=Dummy(),
                               input_data=['localizer_input'],
                               input_steps=[unbinner],
                               adapter={
                                   'y_pred': ([('localizer_unbinner', 'prediction_coordinates')], identity_inputs),
                                   'y_true': ([('localizer_input', 'y')], get_localizer_target_column), },
                               cache_dirpath=config['global']['cache_dirpath'])
    return output


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
                                 transformer=Adjuster(**config['aligner_adjuster']),
                                 input_steps=[unbinner],
                                 input_data=['aligner_input'],
                                 adapter={'crop_coordinates': ([('aligner_input', 'X')],
                                                               get_crop_coordinates),
                                          'prediction_coordinates': (
                                              [('aligner_unbinner', 'prediction_coordinates')], identity_inputs)
                                          },
                                 cache_dirpath=config['global']['cache_dirpath'])

    output = SubstitutableStep(name='aligner_output',
                               transformer=Dummy(),
                               input_data=['aligner_input'],
                               input_steps=[adjuster],
                               adapter={
                                   'y_pred': ([('aligner_adjuster', 'prediction_coordinates')], identity_inputs),
                                   'y_true': ([('aligner_input', 'y')], get_aligner_target_column), },
                               cache_dirpath=config['global']['cache_dirpath'])
    return output


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
                                         transformer=ProbabilityCalibration(**config['classifier_calibrator']),
                                         input_steps=[network],
                                         adapter={
                                             'prediction_proba': (
                                                 [('classifier_network', 'prediction_probability')], exp_transform),
                                         },
                                         cache_dirpath=config['global']['cache_dirpath'])

    output = SubstitutableStep(name='classifier_output',
                               transformer=Dummy(),
                               input_steps=[proba_calibrator, encoder],
                               adapter={
                                   'y_pred': ([('classifier_calibrator', 'prediction_probability')], identity_inputs),
                                   'y_true': ([('classifier_encoder', 'y')], get_classifier_target_column), },
                               cache_dirpath=config['global']['cache_dirpath'])
    return output
