from minerva.utils import copy_resources
from .config import SOLUTION_CONFIG
from .pipelines import solution_pipeline
from .registry import registered_tasks, registered_score
from .trainer import Trainer
from ..backend.task_manager import TaskSolutionParser


def dry_run(sub_problem, train_mode, dev_mode, cloud_mode):
    if cloud_mode:
        copy_resources()

    trainer = Trainer(solution_pipeline, SOLUTION_CONFIG, dev_mode)
    print(train_mode, type(train_mode))
    if train_mode:
        trainer.train()
    _evaluate(trainer)


def submit_task(sub_problem, task_nr, filepath, dev_mode, cloud_mode):
    if cloud_mode:
        copy_resources()

    trainer = Trainer(solution_pipeline, SOLUTION_CONFIG, dev_mode)
    user_task_solution, user_config = _fetch_task_solution(filepath)
    task_handler = registered_tasks[task_nr](trainer)
    new_trainer = task_handler.substitute(user_task_solution, user_config)
    new_trainer.train()
    _evaluate(new_trainer)


def _fetch_task_solution(filepath):
    with TaskSolutionParser(filepath) as task_solution:
        user_solution = task_solution['solution']
        user_config = task_solution['CONFIG']
    return user_solution, user_config


def _evaluate(trainer):
    score_valid, score_test = trainer.evaluate()
    print('\nValidation score is {0:.4f}'.format(score_valid))
    print('Test score is {0:.4f}'.format(score_test))

    if score_test > score_valid + 0.05:
        print('Sorry, your validation split is messed up. Fix it please.')
    else:
        print('That is a solid validation')

        if score_test > registered_score:
            print('Congrats you solved the task!')
        else:
            print('Sorry, but this score is not high enough to pass the task')
