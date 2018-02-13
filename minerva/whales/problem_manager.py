import numpy as np

from minerva.utils import copy_resources, check_inputs, process_config, submit_setup, submit_teardown
from .config import SOLUTION_CONFIG
from .pipelines import localization_pipeline, alignment_pipeline, classification_pipeline
from .tasks import initialize_tasks
from .registry import registered_tasks, registered_scores
from .trainer import Trainer
from ..backend.task_manager import TaskSolutionParser

initialize_tasks()

pipeline_dict = {'localization': localization_pipeline,
                 'alignment': alignment_pipeline,
                 'classification': classification_pipeline}


def dry_run(sub_problem, train_mode, dev_mode, cloud_mode):
    config = process_config(SOLUTION_CONFIG, sub_problem)

    if cloud_mode:
        copy_resources()

    pipeline = pipeline_dict[sub_problem]
    check_inputs(train_mode, config, pipeline)

    trainer = Trainer(pipeline, config, dev_mode, cloud_mode, sub_problem)

    if train_mode:
        trainer.train()
    _evaluate(trainer, sub_problem)


def submit_task(sub_problem, task_nr, filepath, dev_mode, cloud_mode):
    config = process_config(SOLUTION_CONFIG, sub_problem)

    if cloud_mode:
        copy_resources()

    pipeline = pipeline_dict[sub_problem]
    check_inputs(train_mode=False, config=config, pipeline=pipeline)

    submit_config = submit_setup(config)


    trainer = Trainer(pipeline, submit_config, dev_mode, cloud_mode, sub_problem)
    user_task_solution, user_config = _fetch_task_solution(filepath)
    task_handler = registered_tasks[task_nr](trainer)
    new_trainer = task_handler.substitute(user_task_solution, user_config)
    new_trainer.train()
    _evaluate(new_trainer, sub_problem)

    submit_teardown(submit_config)



def _fetch_task_solution(filepath):
    with TaskSolutionParser(filepath) as task_solution:
        user_solution = task_solution.get('solution')
        user_config = task_solution.get('CONFIG')
    return user_solution, user_config


def _evaluate(trainer, sub_problem):
    score_valid, score_test = trainer.evaluate()
    print('\nValidation score is {0:.4f}'.format(score_valid))
    print('Test score is {0:.4f}'.format(score_test))

    if np.abs(score_test - score_valid) > registered_scores[sub_problem]['score_std']:
        print('Sorry, your validation split is messed up. Fix it please.')
    else:
        print('That is a solid validation')

        if score_test < registered_scores[sub_problem]['score']:
            print('Congrats you solved the task!')
        else:
            print('Sorry, but this score is not high enough to pass the task')
