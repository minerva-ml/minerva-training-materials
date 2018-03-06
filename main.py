#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib

import click

from minerva.utils import init_logger, setup_torch_multiprocessing, get_logger, SUBPROBLEM_INFERENCE, \
    get_available_problems

logging = get_logger()
PROBLEMS_CHOICE = click.Choice(get_available_problems())


@click.group()
def action():
    pass


@action.command()
@click.option('-p', '--problem', type=PROBLEMS_CHOICE, help='problem to choose', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
def dry_train(problem, dev_mode):
    dry_run(problem, train_mode=True, dev_mode=dev_mode)


@action.command()
@click.option('-p', '--problem', type=PROBLEMS_CHOICE, help='problem to choose', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
def dry_eval(problem, dev_mode):
    dry_run(problem, train_mode=False, dev_mode=dev_mode)


def dry_run(problem, train_mode, dev_mode):
    if problem == 'whales':
        setup_torch_multiprocessing()

    pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
    sub_problems = SUBPROBLEM_INFERENCE.get(problem, {0: None})
    for sub_problem in list(set(sub_problems.values())):
        if sub_problem:
            logging.info('running: {0}'.format(sub_problem))
        pm.dry_run(sub_problem, train_mode, dev_mode)


@action.command()
@click.option('-p', '--problem', type=PROBLEMS_CHOICE, help='problem to choose', required=True)
@click.option('-t', '--task_nr', type=int, help='task number', required=True)
@click.option('-d', '--dev_mode', help='dev mode on', is_flag=True)
@click.option('-f', '--file_path', type=str, help='file path to task solution')
def submit(problem, task_nr, file_path, dev_mode):
    if file_path is None:
        file_path = 'resources/{}/tasks/task{}.ipynb'.format(problem, task_nr)
    if problem == 'whales':
        setup_torch_multiprocessing()

    sub_problems = SUBPROBLEM_INFERENCE.get(problem, {})
    task_sub_problem = sub_problems.get(task_nr, None)

    pm = importlib.import_module('minerva.{}.problem_manager'.format(problem))
    pm.submit_task(task_sub_problem, task_nr, file_path, dev_mode)


if __name__ == "__main__":
    init_logger()
    action()
