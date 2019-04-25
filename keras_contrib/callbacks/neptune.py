"""Neptune integration for Keras via callback.
"""
from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import Callback

import neptune

class NeptuneLogger(Callback):
    """
    Neptune integration with keras callbacks.
    """

    def __init__(
            self,
            project_qualified_name, experiment_name, api_token=None, **kwargs):
        """
        Construct the NeptuneLogger callback and log in to neptune.
        Neptune experiment creation is delayed until the beginning of training.

        Args:
            project_qualified_name (str): The username and project name with
                which to log in.
            experiment_name (str): The name to give to the new experiment.
            api_token (str, optional): The Neptune API token to use as
                credentials. By default this is read from the environment
                variable NEPTUNE_API_TOKEN.
            **kwargs:
                For full list see neptune.projects.Project.create_experiment.
                Some useful keyword names include: description (str),
                params (dict), properties (dict), tags (list of str),
                upload_source_files (list of str paths).
                Note that the value of this can change between instatiation
                and on_train_begin.

        """
        self.experiment_name = experiment_name
        self.experiment_kwargs = kwargs
        self.project_qualified_name = project_qualified_name
        self.experiment = None

        self.session = neptune.sessions.Session(api_token=api_token)
        self.project = self.session.get_project(project_qualified_name)

    def __del__(self):
        if self.experiment:
            self.experiment.stop()

    def on_train_begin(self, logs):
        if self.experiment:
            return
        self.experiment = self.project.create_experiment(
            self.experiment_name, **self.experiment_kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        for key, value in logs.items():
            try:
                self.experiment.send_metric(key, epoch, float(value))
            except ValueError:
                pass # Ignore non numeric values
