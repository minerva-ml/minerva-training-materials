import os
import subprocess
import sys
from importlib import import_module
import tempfile


class Task():
    def __init__(self, trainer):
        self.trainer = trainer

    def substitute(self, user_solution, user_config):
        self.modify_config(user_config)
        self.modify_pipeline(user_solution, user_config)
        self.modify_trainer(user_solution, user_config)
        return self.trainer

    def modify_trainer(self, user_solution, user_config):
        pass

    def modify_config(self, user_config):
        pass

    def modify_pipeline(self, user_solution, user_config):
        pass


class TaskSolutionParser(tempfile.TemporaryDirectory):
    """Todo:
    exit doesn't work on exceptions and leaves converted .py file out there
    """

    def __init__(self, filepath):
        super().__init__()
        self.filepath = os.path.abspath(filepath)

    def __enter__(self):
        tempdir = super().__enter__()
        cmd = 'mv {} {}'.format(self.filepath, tempdir)
        subprocess.call(cmd, shell=True)
        self.filepath = os.path.join(tempdir, os.path.basename(self.filepath))

        if self.filepath.endswith('.ipynb'):
            cmd = 'jupyter nbconvert --to python {}'.format(self.filepath)
            subprocess.call(cmd, shell=True)
            module_filepath = self.filepath.replace('.ipynb', '.py')
        else:
            module_filepath = self.filepath

        module_dir, module_filename = os.path.split(module_filepath)
        module_name = module_filename.replace('.py', '')
        if module_filename not in os.listdir(module_dir):
            raise ValueError('Failed to convert your solution to pipeline element. Likely problem is indentation format')
        sys.path.append(module_dir)
        task_solution = vars(import_module(module_name))
        return task_solution

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__()
