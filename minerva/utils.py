import sys
import os
import subprocess
import shutil
import logging


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


def is_neptune_cloud():
    PUBLIC_RESOURCES_DIR = '/public/minerva/resources'
    return os.path.exists(PUBLIC_RESOURCES_DIR)


def setup_env(config, sub_problem):
    if is_neptune_cloud():
        config = setup_cloud(config, sub_problem)
        cloud_mode = True
    else:
        config = process_config(config, sub_problem)
        cloud_mode = False
    return config, cloud_mode


def setup_cloud(config, sub_problem):
    PUBLIC_RESOURCES_DIR = '/public/minerva/resources'
    PUBLIC_OUTPUT_DIR = '/output'
    experiment_dir = config['global']['cache_dirpath']

    if experiment_dir.startswith(PUBLIC_RESOURCES_DIR) or experiment_dir == '':
        cmd = 'cp -rf {}/* {}/'.format(PUBLIC_RESOURCES_DIR, PUBLIC_OUTPUT_DIR)
        subprocess.call(cmd, shell=True)
        config = eval(str(config).replace(PUBLIC_RESOURCES_DIR, PUBLIC_OUTPUT_DIR))
    elif experiment_dir.startswith(PUBLIC_OUTPUT_DIR):
        pass
    else:
        raise ValueError('Wrong solution_dir: {} for cloud mode. '
                         'Choose either /public/minerva/resources... if you want to use our solution or /output... if your want to dry_train your solution from scratch'.format(
            experiment_dir))

    experiment_dir_ = config['global']['cache_dirpath']
    if sub_problem is not None:
        if not experiment_dir_.endswith(sub_problem):
            config = eval(str(config).replace(experiment_dir_, os.path.join(experiment_dir_, sub_problem)))

    return config


def check_inputs(train_mode, config, pipeline):
    solution_path = config['global']['cache_dirpath']

    if train_mode:
        if os.path.exists(solution_path):
            if 'transformers' in os.listdir(solution_path):
                transformers_in_dir = set(os.listdir(os.path.join(solution_path, 'transformers')))
                transformers_in_pipeline = set(pipeline(config).all_steps.keys())

                if transformers_in_dir.issuperset(transformers_in_pipeline):
                    missing_transformers = transformers_in_pipeline - transformers_in_dir
                    raise ValueError(
                        """Cannot run dry_train on the solution_dir that contains trained transformers. Perhaps you wanted to run dry_eval?""".format(
                            solution_path, list(missing_transformers)))

    else:
        if os.path.exists(solution_path):
            if 'transformers' not in os.listdir(solution_path):
                raise ValueError(
                    """Specified solution_dir {} is missing 'transformers' directory. Use dry_train or specify the path to trained pipeline
                    """.format(solution_path))
            else:
                transformers_in_dir = set(os.listdir(os.path.join(solution_path, 'transformers')))
                transformers_in_pipeline = set(pipeline(config).all_steps.keys())

                if not transformers_in_dir.issuperset(transformers_in_pipeline):
                    missing_transformers = transformers_in_pipeline - transformers_in_dir
                    raise ValueError(
                        """Specified solution_dir {} is missing trained transformers: {}. Use dry_train or specify the path to trained pipeline""".format(
                            solution_path, list(missing_transformers)))
        else:
            raise ValueError(
                """Specified solution_dir {} doesn't exist. Use dry_train or specify the path to trained pipeline
                """.format(solution_path))


def process_config(config, sub_problem):
    if sub_problem is not None:
        experiment_dir = config['global']['cache_dirpath']
        if not experiment_dir.endswith(sub_problem):
            config = eval(str(config).replace(experiment_dir, os.path.join(experiment_dir, sub_problem)))
    return config


def submit_setup(config):
    experiment_dir = config['global']['cache_dirpath']
    submission_dir = os.path.join(experiment_dir, 'submit_solution')

    create_clean_dir(submission_dir)
    copytree(experiment_dir, submission_dir)
    config = eval(str(config).replace(experiment_dir, submission_dir))
    return config


def submit_teardown(config):
    experiment_dir = config['global']['cache_dirpath']

    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)


def create_clean_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


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
