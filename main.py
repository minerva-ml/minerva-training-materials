import importlib

import click

from minerva.utils import init_logger, setup_torch_multiprocessing, get_logger, SUBPROBLEM_INFERENCE

logging = get_logger()


@click.group()
def action():
    pass


@action.command()
@click.option('-p', '--problem', help='problem to choose', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
def dry_train(problem, dev_mode):
    dry_run(problem, train_mode=True, dev_mode=dev_mode)


@action.command()
@click.option('-p', '--problem', help='problem to choose', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
def dry_eval(problem, dev_mode):
    dry_run(problem, train_mode=False, dev_mode=dev_mode)


def dry_run(problem, train_mode, dev_mode):
    if problem == 'whales':
        setup_torch_multiprocessing()

    subproblems = SUBPROBLEM_INFERENCE.get(problem)
    if subproblems:
        for sub_problem in list(set(subproblems.values())):
            pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
            logging.info('running: {0}'.format(sub_problem))
            pm.dry_run(sub_problem, train_mode, dev_mode)
    else:
        pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
        sub_problem = None
        pm.dry_run(sub_problem, train_mode, dev_mode)


@action.command()
@click.option('-p', '--problem', help='problem to choose', required=True)
@click.option('-t', '--task_nr', help='task number', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
@click.option('-f', '--filepath', type=str, help='filepath_to_solution')
def submit(problem, task_nr, filepath, dev_mode):
    if filepath is None:
        filepath = 'resources/{}/tasks/task{}.ipynb'.format(problem, task_nr)
    if problem == 'whales':
        setup_torch_multiprocessing()

    task_nr = int(task_nr)
    subproblems = SUBPROBLEM_INFERENCE.get(problem)
    if subproblems:
        task_subproblem = subproblems.get(task_nr)
    else:
        task_subproblem = None

    pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
    pm.submit_task(task_subproblem, task_nr, filepath, dev_mode, cloud_mode)


if __name__ == "__main__":
    init_logger()
    action()
