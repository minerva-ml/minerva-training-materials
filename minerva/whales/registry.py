registered_scores = {'localization': {'score': 110, 'score_std': 20},
                     'alignment': {'score': 200, 'score_std': 20},
                     'classification': {'score': 1.3, 'score_std': 0.2},
                     'end_to_end': {'score': 1.3, 'score_std': 0.2}
                     }
registered_tasks = {}


def register_task(func):
    func_name = func.__name__
    id_ = int(func_name.replace('Task', ''))
    registered_tasks[id_] = func
    return func
