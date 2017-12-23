import importlib

import click

from minerva.utils import init_logger, setup_torch_multiprocessing, get_logger

logging = get_logger()


@click.group()
def action():
    pass


@action.command()
@click.option('-p', '--problem', help='problem to choose', required=True)
@click.option('-s', '--sub_problem', help='sub problem to choose', required=False)
@click.option('-e', '--eval_mode', help='evaluate only mode', default=True, required=False)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
@click.option('-c', '--cloud_mode', help='cloud mode on', is_flag=True)
def dry_run(problem, sub_problem, eval_mode, dev_mode, cloud_mode):
    if problem == 'whales':
        setup_torch_multiprocessing()

    pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
    logging.info('running: {0}'.format(sub_problem))
    pm.dry_run(sub_problem, eval_mode, dev_mode, cloud_mode)


@action.command()
@click.option('-s', '--sub_problem', help='sub, problem to choose', required=False)
@click.option('-p', '--problem', help='problem to choose', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
@click.option('-t', '--task_nr', default=1, help='task number')
@click.option('-f', '--filepath', type=str, help='filepath_to_solution')
@click.option('-c', '--cloud_mode', help='cloud mode on', is_flag=True)
def submit(problem, sub_problem, task_nr, filepath, dev_mode, cloud_mode):
    if problem == 'whales':
        setup_torch_multiprocessing()
    pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
    pm.submit_task(sub_problem, task_nr, filepath, dev_mode, cloud_mode)


if __name__ == "__main__":
    init_logger()
    action()
