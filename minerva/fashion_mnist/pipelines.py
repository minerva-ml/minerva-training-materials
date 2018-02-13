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
