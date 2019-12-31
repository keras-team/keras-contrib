import random
import itertools as it
import os
import uuid
import tempfile
import shutil
from keras.models import load_model, save_model
from keras.callbacks import CallbackList
import keras.backend as K


def create_random_configurations(search_space, count=5):
    """
    Generate a list of hyperparameter configurations sampled randomly from the provided search space.

    # Arguments
        search_space: Dictionary of distributions to sample from. Keys are hyperparameter names, values are
                      lists or scipy distributions to sample from. Lists are sampled from uniformly.
        count: The number of estimators to sample.

    # Returns
        A list of dictionaries. Each dictionary represents a configuration of
        hyperparameters, mapping {'parameter_name':value}.
    """
    configuration_list = []
    for i in range(count):
        configuration = {}
        for k, v in search_space.items():
            if type(v) is list:
                configuration[k] = random.choice(v)
            else:
                configuration[k] = float(v.rvs())
        configuration_list.append(configuration)

    return configuration_list


def create_grid_configurations(search_space):
    """
    Generate a grid of hyperparameter configurations exhaustively over a search space.

    # Arguments
        search_space: Dictionary of lists to generate configurations exhaustively from.
                    Keys are hyperparameter names, values are lists of hyperparameter values.

    # Returns
        A list of dictionaries. Each dictionary represents a configuration of
        hyperparameters, mapping {'parameter_name':value}.

    """
    for k, v in search_space.items():
        if type(v) is not list:
            raise (Exception('Grid search space must be a dictionary containing only lists'))
    configuration_list = it.product(*(search_space[p] for p in search_space.keys()))
    configuration_list = [{k: v for k, v in zip(search_space.keys(), conf)} for conf in configuration_list]

    return configuration_list


class SearchWrapper:
    def __init__(self, build_fn, callbacks=None, model_cache_dir=None, **hyperparameters):
        """

        A wrapper around Sequential() or Model() models that only keeps models in memory when they are being fitted.
        This is useful for cases like hyperparameter optimization, where you might want to keep several large models
        post-search to refit on the entire dataset.

        Additionally, allows for scikit-learn style partial fitting, while still maintaining callbacks.
        Note that train_on_batch() itself does not manage callbacks, so this wrapper takes care of it.
        This is useful for future proofing for search techniques that rely on adaptive epoch allocation, such as
        Hyperband or Harmonica search. These techniques often decide based on the initial few batches whether to
        continue training - thus need more granularity than epoch-level training.

        Models can be saved to disk, but callbacks cannot - so we keep callbacks in memory so training can be resumed.

        Important note: This wrapper calls K.clear_session() when using tensorflow - this is required as compilation
        time will crawl to a halt if the computational graph is not cleared between configuration evaluations.

        # Arguments
            build_fn: callable function
                    A build function, which takes hyperparameters as kwargs, and returns a compiled Sequential() or
                    Model() model using those hyperparameters.
            callbacks: Callbacks to apply to all models.
            model_cache_dir: Location to keep model when being stored out-of-memory. By default, model is saved to a
                            temporary directory.
            **hyperparameters: Additional parameters to pass to the Sequential model when it is built.
        """

        # We want to be able to save and load models from disk and continue training without interruption.
        # Since callbacks need to stay in memory, manage it ourselves:
        self.batches_trained = 0
        self.epochs = 0

        # What model to build, and what hyperparameters to build it with:
        self.build_fn = build_fn
        self.hyperparameters = hyperparameters

        self.steps_per_epoch = None
        self.model = None
        self.initialized = False
        self.partial_fit_initialized = False
        self.id = str(uuid.uuid4())
        self.tmp_dir = False

        if model_cache_dir:
            self.model_cache_dir = model_cache_dir
        else:
            self.model_cache_dir = tempfile.mkdtemp()
            self.tmp_dir = True

        # Callbacks need to persist even when model is dumped to disk
        self.callbacks = callbacks
        if self.callbacks:
            self.callback_list = CallbackList(self.callbacks)
        else:
            self.callbacks = []
            self.callback_list = CallbackList(self.callbacks)

        # Create a cache so we can keep models on disk.
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)

    def fit(self, X, y, **fit_params):
        """
        Fit the model on an entire training dataset.

        Args:
            X: Numpy array, or list of arrays if using Model(). Training data set.
            y: Numpy array, or list of arrays if using Model(). Corresponding labels for training data.
            **fit_params: Additional parameters to pass to .fit() call.
        Returns:
            The History callback of a trained model.
        """
        self.maybe_load_model()

        history = self.model.fit(X, y, callbacks=self.callbacks, **fit_params)

        self.save_and_clear_model()
        return history

    def fit_generator(self, generator, steps_per_epoch, **fit_params):
        """
        Fit the model on an entire training dataset.

        Args:
            generator: A generator. The output of the generator must be either
                a tuple (inputs, targets)
                a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples. The generator is expected to loop over its data
                indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.

            steps_per_epoch:  Total number of steps (batches of samples) to yield from generator before declaring one
                            epoch finished and starting the next epoch. It should typically be equal to the number of
                            unique samples of your dataset divided by the batch size.
            **fit_params: Additional parameters to pass to .fit() call.
        Returns:
            The History callback of a trained model.
        """
        self.maybe_load_model()

        history = self.model.fit_generator(generator, steps_per_epoch, callbacks=self.callbacks, **fit_params)

        self.save_and_clear_model()
        return history

    def save_and_clear_model(self):
        """
        Save model to disk and remove it from memory.
        """
        if self.model is not None:
            save_model(self.model, os.path.join(self.model_cache_dir, self.id + ".h5"), overwrite=True)
            self.model = None

            if K._BACKEND == 'tensorflow':
                K.clear_session()

    def maybe_load_model(self):
        """
        If model is not loaded, load model from disk. If model has not been initialized, initialize it.
        """
        if not self.initialized:
            self.model = self.build_fn(**self.hyperparameters)
            self.save_and_clear_model()
            self.initialized = True

        if not self.model:
            self.model = load_model(os.path.join(self.model_cache_dir, self.id + ".h5"))
            self.callback_list.set_model(self.model)

    def start_train_partial(self, steps_per_epoch):
        """
        Must be called at the start of partial_fit training. Initializes callbacks for training process.
        """
        self.maybe_load_model()

        self.callback_list.set_model(self.model)

        self.callback_list.on_train_begin()
        self.callback_list.on_epoch_begin(self.epochs)

        self.steps_per_epoch = steps_per_epoch
        self.partial_fit_initialized = True

    def next_epoch(self):
        """
        Called after every partial_fit epoch to update the callbacks list.
        """
        self.callback_list.on_epoch_end(self.epochs)
        self.epochs += 1
        self.callback_list.on_epoch_begin(self.epochs)

    def end_train_partial(self):
        """
        Called at the end of partial_fit training to finalize callbacks, and save and clear model from memory.

        You really only ever need to call this if you are finalizing a model when it is in the middle of an epoch.
        Otherwise save_and_clear_model() is used to cache a model if you intend on resuming partial training.
        """
        self.callback_list.on_epoch_end(self.epochs)
        self.callback_list.on_train_end()
        self.save_and_clear_model()

    def partial_fit(self, X, y, **fit_params):
        """
        Perform training on a single batch, and update callbacks accordingly.
        Args:
            X: Numpy array, or list of arrays if using Model(), the training batch.
            y: Numpy array, or list of arrays if using Model(), the labels for that batch.
            **fit_params: Additional parameters to be passed to train_on_batch.
        Returns:
            Loss of model after training on batch.
        """
        assert self.partial_fit_initialized, 'Partial fitting not initialized, please call start_train_partial() first'

        self.maybe_load_model()

        self.callback_list.on_batch_begin(self.batches_trained)

        loss = self.model.train_on_batch(X, y, **fit_params)

        self.callback_list.on_batch_end(self.batches_trained)
        self.batches_trained += 1

        if self.batches_trained % self.steps_per_epoch == 0:
            self.next_epoch()

        return loss

    def evaluate(self, X, y, **fit_params):
        """
        Args:
            X: Numpy array, or list of arrays if using Model(), the data to evaluate on.
            y: Numpy array, or list of arrays if using Model(), the labels for that data.
            **fit_params: Additional parameters to be passed to evaluate.
        Returns:
            List. Evaluation metrics from model evaluated on test set.
        """
        self.maybe_load_model()

        result = self.model.evaluate(X, y, **fit_params)

        self.save_and_clear_model()
        return result

    def evaluate_generator(self, generator, steps, **fit_params):
        """
        Args:
            generator: A generator. The output of the generator must be a tuple (inputs, targets).
            steps:  An integer. Total number of steps (batches of samples) to yield from generator to validate against.
            **fit_params: Additional parameters to be passed to evaluate_generator.
        Returns:
            List. Evaluation metrics from model evaluated on test set.
        """
        self.maybe_load_model()

        result = self.model.evaluate_generator(generator, steps, **fit_params)

        self.save_and_clear_model()
        return result

    def __del__(self):
        if self.tmp_dir:
            shutil.rmtree(self.model_cache_dir)
