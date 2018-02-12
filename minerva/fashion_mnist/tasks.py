from .registry import register_task
from ..backend.task_manager import Task


def initialize_tasks():
    pass


@register_task
class Task1(Task):
    def modify_config(self, user_config):
        self.trainer.config['model']['architecture_config']['model_params'] = user_config
        return self

    def modify_pipeline(self, user_solution, user_config):
        self.trainer.pipeline.get_step('keras_model').transformer._build_model = user_solution
        self.trainer.pipeline.get_step('keras_model').transformer.model = self.trainer.pipeline.get_step(
            'keras_model').transformer._compile_model(**self.trainer.config['model']['architecture_config'])
        self.trainer.pipeline.get_step('keras_model').is_substituted = True
        return self


@register_task
class Task2(Task):
    def modify_config(self, user_config):
        self.trainer.config['model']['architecture_config']['optimizer_params']['lr'] = user_config['lr']
        self.trainer.config['model']['architecture_config']['optimizer_params']['momentum'] = user_config['momentum']
        self.trainer.config['loader']['augmentation']['train']['flow']['batch_size'] = user_config['batch_size']
        return self

    def modify_pipeline(self, user_solution, user_config):
        self.trainer.pipeline.get_step('keras_model').transformer.model = self.trainer.pipeline.get_step(
            'keras_model').transformer._compile_model(**self.trainer.config['model']['architecture_config'])
        self.trainer.pipeline.get_step('keras_model').is_substituted = True
        return self


@register_task
class Task3(Task):
    def modify_config(self, user_config):
        self.trainer.config['loader']['augmentation']['train'] = user_config
        return self

    def modify_pipeline(self, user_solution, user_config):
        self.trainer.pipeline.get_step('loader').transformer.datagen_builder = user_solution
        self.trainer.pipeline.get_step('loader').is_substituted = True
        return self


@register_task
class Task4(Task):
    def modify_trainer(self, user_solution, user_config):
        self.trainer.cross_validation_split = user_solution
        self.trainer.pipeline.get_step('loader').is_substituted = True
        return self
