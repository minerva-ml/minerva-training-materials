from .models import SimpleClassifier
from .preprocessing import KerasDataLoader
from ..backend.base import SubstitutableStep, stack_inputs, sum_inputs
from ..backend.postprocessing import ClassPredictor, PredictionAverage


def solution_pipeline(config):
    loader = SubstitutableStep(name='input',
                               transformer=KerasDataLoader(**config['loader']),
                               input_data=['data'],
                               cache_dirpath=config['global']['cache_dirpath'])
    model = SubstitutableStep(name='keras_model',
                              transformer=SimpleClassifier(**config['model']),
                              input_steps=[loader],
                              cache_dirpath=config['global']['cache_dirpath']
                              )
    output = SubstitutableStep(name='class_predictions',
                               transformer=ClassPredictor(),
                               input_steps=[model],
                               cache_dirpath=config['global']['cache_dirpath'])
    return output


def complicated_pipeline(config):
    loader = SubstitutableStep(name='crop_input',
                               transformer=KerasDataLoader(**config['loader']),
                               input_data=['data'],
                               cache_dirpath=config['global']['cache_dirpath'])
    model1 = SubstitutableStep(name='keras_model1',
                               transformer=SimpleClassifier(**config['model']),
                               input_steps=[loader],
                               cache_dirpath=config['global']['cache_dirpath'])
    model2 = SubstitutableStep(name='keras_model2',
                               transformer=SimpleClassifier(**config['model']),
                               input_steps=[loader],
                               cache_dirpath=config['global']['cache_dirpath'])
    model3 = SubstitutableStep(name='keras_model3',
                               transformer=SimpleClassifier(**config['model']),
                               input_steps=[loader],
                               cache_dirpath=config['global']['cache_dirpath'])
    prediction_proba1 = SubstitutableStep(name='average_prediction1',
                                          transformer=PredictionAverage(),
                                          input_steps=[model1, model2, model3],
                                          adapter={'prediction_proba_list': (
                                              [('keras_model1', 'prediction_proba'),
                                               ('keras_model2', 'prediction_proba'),
                                               ('keras_model3', 'prediction_proba')], stack_inputs)},
                                          cache_dirpath=config['global']['cache_dirpath'])

    loader1 = SubstitutableStep(name='clf_input',
                                transformer=KerasDataLoader(**config['loader']),
                                input_data=['data'],
                                cache_dirpath=config['global']['cache_dirpath'])
    model4 = SubstitutableStep(name='keras_model4',
                               transformer=SimpleClassifier(**config['model']),
                               input_steps=[loader1],
                               cache_dirpath=config['global']['cache_dirpath'])
    model5 = SubstitutableStep(name='keras_model5',
                               transformer=SimpleClassifier(**config['model']),
                               input_steps=[loader1],
                               cache_dirpath=config['global']['cache_dirpath'])
    model6 = SubstitutableStep(name='keras_model6',
                               transformer=SimpleClassifier(**config['model']),
                               input_steps=[loader1],
                               cache_dirpath=config['global']['cache_dirpath'])
    prediction_proba2 = SubstitutableStep(name='average_prediction2',
                                          transformer=PredictionAverage(),
                                          input_steps=[model4, model5, model6],
                                          adapter={'prediction_proba_list': (
                                              [('keras_model4', 'prediction_proba'),
                                               ('keras_model5', 'prediction_proba'),
                                               ('keras_model6', 'prediction_proba')], stack_inputs)},
                                          cache_dirpath=config['global']['cache_dirpath'])

    output = SubstitutableStep(name='sum_prediction',
                               transformer=ClassPredictor(),
                               input_steps=[prediction_proba1, prediction_proba2],
                               adapter={'prediction_proba': (
                                   [('average_prediction1', 'prediction_proba'),
                                    ('average_prediction2', 'prediction_proba')], sum_inputs)},
                               cache_dirpath=config['global']['cache_dirpath'])
    return output
