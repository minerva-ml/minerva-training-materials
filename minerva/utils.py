import logging
import os
import subprocess
import sys


def setup_torch_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')


def _welcome_message(fn):
    def wrapper():
        fn()
        get_logger().info('starting experiment...')

    return wrapper


@_welcome_message
def init_logger():
    logger = logging.getLogger('minerva')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)
    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)


def get_logger():
    return logging.getLogger('minerva')


def copy_resources():
    cmd = 'cp -rf /public/minerva/resources /output'
    subprocess.call(cmd, shell=True)


def handle_empty_solution_dir(train_mode, config, pipeline):
    if not train_mode:
        solution_path = config['global']['cache_dirpath']
        if 'transformers' not in os.listdir(solution_path):
            raise ValueError(
                """Specified solution_dir {} is missing 'transformers' directory. Use dry_run with train_mode=True or specify the path to trained pipeline
                """.format(solution_path))
        else:
            transformers_in_dir = set(os.listdir(os.path.join(solution_path, 'transformers')))
            transformers_in_pipeline = set(pipeline(config).all_steps.keys())

            if not transformers_in_dir.issuperset(transformers_in_pipeline):
                missing_transformers = transformers_in_pipeline - transformers_in_dir
                raise ValueError(
                    """Specified solution_dir {} is missing trained transformers: {}. Use dry_run with train_mode=True or specify the path to trained pipeline""".format(
                        solution_path, list(missing_transformers)))


def handle_dry_train(train_mode, config, pipeline):
    if train_mode:
        solution_path = config['global']['cache_dirpath']
        if 'transformers' in os.listdir(solution_path):
            transformers_in_dir = set(os.listdir(os.path.join(solution_path, 'transformers')))
            transformers_in_pipeline = set(pipeline(config).all_steps.keys())

            if transformers_in_dir.issuperset(transformers_in_pipeline):
                missing_transformers = transformers_in_pipeline - transformers_in_dir
                raise ValueError(
                    """Cannot run dry_train on the solution_dir that contains trained transformers. Perhaps you wanted to run dry_eval?""".format(
                        solution_path, list(missing_transformers)))


def process_config(solution_config, global_config, sub_problem):
    config = solution_config
    experimet_dir = global_config['exp_root']
    if not experimet_dir.endswith(sub_problem):
        config = eval(str(config).replace(experimet_dir, os.path.join(experimet_dir, sub_problem)))
    return config


SUBPROBLEM_INFERENCE = {'whales': {1: 'localization',
                                   2: 'alignment',
                                   3: 'classification',
                                   4: 'localization',
                                   5: 'localization',
                                   6: 'classification',
                                   7: 'localization',
                                   8: 'alignment',
                                   9: 'classification',
                                   }
                        }
