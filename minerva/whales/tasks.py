from .registry import register_task
from ..backend.task_manager import Task


@register_task
class Task1(Task):
    """
    Note:
        Localizer Architecture
    """

    def modify_config(self, user_config):
        self.trainer.config['localizer_network']['architecture_config']['model_params'] = user_config
        return self

    def modify_pipeline(self, user_solution, user_config):
        self.trainer.pipeline.get_step('localizer_network').transformer._build_model = user_solution
        self.trainer.pipeline.get_step('localizer_network').is_substituted = True
        return self


@register_task
class Task2(Task):
    """
    Note:
        Localizer weight regularization
    """

    def modify_config(self, user_config):
        self.trainer.config['localizer_network']['architecture_config']['regularizer_params'] = user_config
        return self

    def modify_pipeline(self, user_solution, user_config):
        self.trainer.pipeline.get_step('localizer_network').transformer.weight_regularization = user_solution
        self.trainer.pipeline.get_step('localizer_network').is_substituted = True
        return self
