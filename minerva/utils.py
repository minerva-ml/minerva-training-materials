import sys
import subprocess
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

    # # setup file handler and message format
    # log_file = GLOBAL_CONFIG['log_file']
    # fh_tr = logging.FileHandler(filename=log_file, mode='w')  # fh_training.close()
    # fh_tr.setLevel(logging.INFO)
    # fh_tr.setFormatter(fmt=message_format)
    # logger.addHandler(fh_tr)

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
