registered_score = 0.9
registered_tasks = {}


def register_task(func):
    func_name = func.__name__
    id_ = int(func_name.replace('Task', ''))
    registered_tasks[id_] = func
    return func

