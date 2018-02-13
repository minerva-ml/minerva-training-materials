import os
import pprint

import numpy as np
from sklearn.externals import joblib

from minerva.utils import get_logger
from .utils import view_graph, plot_graph

logger = get_logger()


class Step:
    def __init__(self, name, transformer, input_steps=[], input_data=[], adapter=None, cache_dirpath=None,
                 save_outputs=[], save_graph=False):
        self.name = name
        self.transformer = transformer

        self.input_steps = input_steps
        self.input_data = input_data
        self.adapter = adapter

        if save_graph:
            self._save_graph()

        self.cache_dirpath = cache_dirpath
        self._prep_cache(cache_dirpath, save_outputs)

    def _prep_cache(self, cache_dirpath, save_outputs):
        for dirname in ['transformers', 'outputs']:
            os.makedirs(os.path.join(cache_dirpath, dirname), exist_ok=True)

        self.cache_dirpath_transformers = os.path.join(cache_dirpath, 'transformers')
        self.save_dirpath_outputs = os.path.join(cache_dirpath, 'outputs')

        self.cache_filepath_step_transformer = os.path.join(self.cache_dirpath_transformers, self.name)

        save_output_filenames = {}
        for save_output in save_outputs:
            save_output_filenames[save_output] = os.path.join(self.save_dirpath_outputs,
                                                              '{}_{}'.format(self.name, save_output))
        self.save_filepath_step_outputs = save_output_filenames

    def _save_graph(self):
        graph_filepath = os.path.join(self.cache_dirpath, '{}_graph.json'.format(self.name))
        logger.info('Saving graph to {}'.format(graph_filepath))
        joblib.dump(self.graph_info, graph_filepath)

    @property
    def named_steps(self):
        return {step.name: step for step in self.input_steps}

    def get_step(self, name):
        return self.all_steps[name]

    @property
    def is_cached(self):
        return os.path.exists(self.cache_filepath_step_transformer)

    @property
    def _can_load_fit_transform(self):
        return self.is_cached

    @property
    def _can_load_transform(self):
        return self.is_cached

    def fit_transform(self, data):
        step_inputs = {}
        if self.input_data is not None:
            for input_data_part in self.input_data:
                step_inputs[input_data_part] = data[input_data_part]

        for input_step in self.input_steps:
            step_inputs[input_step.name] = input_step.fit_transform(data)

        if self.adapter:
            step_inputs = self.adapt(step_inputs)
        else:
            step_inputs = self.unpack(step_inputs)
        step_output_data = self._cached_fit_transform(step_inputs)
        return step_output_data

    def _cached_fit_transform(self, step_inputs):
        if self._can_load_fit_transform:
            logger.info('step {} loading...'.format(self.name))
            self.transformer.load(self.cache_filepath_step_transformer)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            step_output_data = self.transformer.fit_transform(**step_inputs)
            logger.info('step {} saving transformer...'.format(self.name))
            self.transformer.save(self.cache_filepath_step_transformer)
            logger.info('step {} saving outputs...'.format(self.name))
            self._save_selected_outputs(step_output_data)
        return step_output_data

    def _save_selected_outputs(self, output_data):
        for name, filepath in self.save_filepath_step_outputs.items():
            joblib.dump(output_data[name], filepath)

    def transform(self, data):
        step_inputs = {}
        if self.input_data is not None:
            for input_data_part in self.input_data:
                step_inputs[input_data_part] = data[input_data_part]

        for input_step in self.input_steps:
            step_inputs[input_step.name] = input_step.transform(data)

        if self.adapter:
            step_inputs = self.adapt(step_inputs)
        else:
            step_inputs = self.unpack(step_inputs)
        step_output_data = self._cached_transform(step_inputs)
        return step_output_data

    def _cached_transform(self, step_inputs):
        if self._can_load_transform:
            logger.info('step {} loading...'.format(self.name))
            self.transformer.load(self.cache_filepath_step_transformer)
            logger.info('step {} transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)
        else:
            raise ValueError('No transformer cached {} in {}'.format(self.name, self.cache_filepath_step_transformer))
        return step_output_data

    def adapt(self, step_inputs):
        logger.info('step {} adapting inputs'.format(self.name))
        adapted_steps = {}
        for adapted_name, mapping in self.adapter.items():
            if isinstance(mapping, str):
                adapted_steps[adapted_name] = step_inputs[mapping]
            else:
                (step_mapping, func) = mapping
                raw_inputs = [step_inputs[step_name][step_var] for step_name, step_var in step_mapping]
                adapted_steps[adapted_name] = func(raw_inputs)
        return adapted_steps

    def unpack(self, step_inputs):
        logger.info('step {} unpacking inputs'.format(self.name))
        unpacked_steps = {}
        for step_name, step_dict in step_inputs.items():
            unpacked_steps = {**unpacked_steps, **step_dict}
        return unpacked_steps

    @property
    def all_steps(self):
        all_steps = {}
        all_steps = self._get_steps(all_steps)
        return all_steps

    def _get_steps(self, all_steps):
        for input_step in self.input_steps:
            all_steps = input_step._get_steps(all_steps)
        all_steps[self.name] = self
        return all_steps

    @property
    def graph_info(self):
        graph_info = {'edges': set(),
                      'nodes': set()}

        graph_info = self._get_graph_info(graph_info)

        return graph_info

    def _get_graph_info(self, graph_info):
        for input_step in self.input_steps:
            graph_info = input_step._get_graph_info(graph_info)
            graph_info['edges'].add((input_step.name, self.name))
        graph_info['nodes'].add(self.name)
        for input_data in self.input_data:
            graph_info['nodes'].add(input_data)
            graph_info['edges'].add((input_data, self.name))
        return graph_info

    def plot_graph(self, filepath):
        plot_graph(self.graph_info, filepath)

    def __str__(self):
        return pprint.pformat(self.graph_info)

    def _repr_html_(self):
        return view_graph(self.graph_info)


class SubstitutableStep(Step):
    def __init__(self, is_substituted=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_substituted = is_substituted

    @property
    def input_step_is_substituted(self):
        return any(input_step.is_substituted for input_step in self.input_steps) # Todo this is wrong only input steps not recursive

    @property
    def _can_load_fit_transform(self):
        return self.is_cached and not self.is_substituted and not self.input_step_is_substituted

    @property
    def _can_load_transform(self):
        return self.is_cached

class BaseTransformer:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return NotImplementedError

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def load(self, filepath):
        return self

    def save(self, filepath):
        pass


class Output(BaseTransformer):
    def transform(self, **kwargs):
        return kwargs

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class MockTransformer(BaseTransformer):
    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def load(self, filepath):
        return self

    def save(self, filepath):
        pass


class Dummy(BaseTransformer):
    def transform(self, **kwargs):
        return kwargs

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


def identity_inputs(inputs):
    return inputs[0]


def stack_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return stacked


def sum_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return np.sum(stacked, axis=0)


def average_inputs(inputs):
    stacked = np.stack(inputs, axis=0)
    return np.mean(stacked, axis=0)


def exp_transform(inputs):
    return np.exp(inputs[0])
