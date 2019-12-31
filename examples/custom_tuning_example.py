"""
A small example of hyperparameter search in Keras.

First, try grid search.
Then, try a wide random search (many configs) with only a few batches of data - a 'short random search'
Use the best model from 'short random', train to completion, compare with grid results.

The key here is that we only keep one model in memory at once! Large searches (and large models) mean models have to be
kept on disk, at least when not being trained, and must be easy to resume training - callbacks and everything.

You'll probably want a GPU for this, it takes about 8-9 mins on a 980ti.

"""
from __future__ import print_function

from scipy.stats.distributions import uniform
from time import time
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras_contrib.utils.tuning import create_grid_configurations, create_random_configurations, SearchWrapper


# Simple callback to demonstrate that callback state is preserved even when model is dumped to disk.
class ExampleCallback(keras.callbacks.Callback):
    def __init__(self):
        super(ExampleCallback, self).__init__()
        self.total_batches = 0

    def on_train_begin(self, logs=None):
        print("\tTraining Started!")

    def on_batch_end(self, batch, logs=None):
        self.total_batches += 1

    def on_epoch_end(self, epoch, logs=None):
        print("\t\tEpoch", epoch, "completed.", self.total_batches, "batches total.")


# Setup:
batch_size = 100  # Make everything nice round numbers to make logs easy to read
epochs = 10

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


# First we define a function that takes in hyperparameters (you must give them defaults), and returns a compiled model.
def build_fn(activation_hyperparam='relu', kernel_count_hyperparam=16, dense_unit_hyperparam=16,
             optimizer_hyperparam='adam', dropout_hyperparam=0.5):
    nn = Sequential()

    nn.add(BatchNormalization(input_shape=X_train[0].shape))
    nn.add(Conv2D(kernel_count_hyperparam, 3, activation=activation_hyperparam))
    nn.add(Dropout(dropout_hyperparam))
    nn.add(Conv2D(kernel_count_hyperparam, 3, activation=activation_hyperparam))
    nn.add(MaxPooling2D())
    nn.add(Conv2D(kernel_count_hyperparam, 3, activation=activation_hyperparam))
    nn.add(Flatten())
    nn.add(Dense(dense_unit_hyperparam, activation=activation_hyperparam))
    nn.add(Dense(10, activation='softmax'))

    nn.compile(optimizer=optimizer_hyperparam, loss='categorical_crossentropy', metrics=['accuracy'])

    return nn


# --- Simple grid search example ---
# Now we define using a dictionary the space of hyperparameters we want to search.
# Note that we don't have to search all of the specified hyperparameters in build_fn
#   (build_fn will just use the defaults you specified above)
# Also note that every value must be a list, even if it is a list of a single item

start_grid = time()
print("GRID SEARCH:")

search_space = {
    'activation_hyperparam': ['relu', 'sigmoid'],
    'kernel_count_hyperparam': [8],
    'dense_unit_hyperparam': [1, 16, 32],
    'dropout_hyperparam': [0., .5]
}

# Create an exhaustive list of dictionaries, each dictionary representing a hyperparameter config
configurations = create_grid_configurations(search_space)

best_grid_score = 0
best_grid_model = None
for hyperparameters in configurations:
    cbk = ExampleCallback()
    model = SearchWrapper(build_fn, callbacks=[cbk],
                          **hyperparameters)
    # SearchWrapper takes a build function, your callbacks that you would normally pass to fit,
    # and your **hyperparameters to instantiate the model with

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)[-1]

    if score > best_grid_score:  # Keep best model
        best_grid_score = score
        best_grid_model = model
    print("\tScore:", score, "Configuration:", hyperparameters, "Time:", int(time() - start_grid), 'seconds')

end_grid = time() - start_grid

# --- Random Search, a common use case: ---
# We want to test a ton of configurations, but only for a few batches each - because many will diverge or fail fast.
# Once we have pruned down to single good configuration (or more if you want), train them until completion.
# Early success is a good indicator of late success in models many times!

# Note we will use random search, so long lists and scipy distributions both work well
start_random = time()
print("RANDOM SEARCH:")
print('Train a bunch of randomly sampled neural networks, but only for a few minibatches')
search_space = {
    'activation_hyperparam': ['relu', 'sigmoid'],
    'kernel_count_hyperparam': [8],
    'dense_unit_hyperparam': list(range(1, 32)),
    'dropout_hyperparam': uniform(0, .5)
}

nb_random_configs = 50  # Number of random configs to try
first_round_batches = 20
validation_size = 1000  # Samples
epoch_batches = int(X_train.shape[0] / batch_size)
total_training_batches = epoch_batches * epochs

# Sample a list of dictionaries, each dictionary representing a hyperparameter config

configurations = create_random_configurations(search_space, nb_random_configs)

# Partial fitting requires us to pass in batch by batch - so we use a generator:
gen = ImageDataGenerator().flow(X_train, y_train, batch_size=batch_size)

print('Training', nb_random_configs, 'random configs for', first_round_batches, 'batches each')
best_random_score = 0
best_random_model = None
for hyperparameters in configurations:
    cbk = ExampleCallback()
    model = SearchWrapper(build_fn, callbacks=[cbk], **hyperparameters)

    # Partial training w/ callbacks requires knowledge of how many batches there are in an epoch:
    model.start_train_partial(steps_per_epoch=epoch_batches)

    # Train for 'first_round_batches' batches:
    for _ in range(first_round_batches):
        model.partial_fit(*next(gen))

    # Evaluate after those batches - we are still at a fraction of an epoch:
    score = model.evaluate(X_test[:validation_size], y_test[:validation_size], verbose=0)[-1]

    model.save_and_clear_model()  # Save and unload model
    if score > best_random_score:
        best_random_score = score
        best_random_model = model
    print("\tScore:", score, "Configuration:", hyperparameters, "Time:", int(time() - start_random), 'seconds')

print('Now, training single best random model from prior step to completion:',
      total_training_batches - first_round_batches, 'additional batches')

# Finish training up to 10 epochs and compare with grid search:
for i in range(total_training_batches - first_round_batches):
    best_random_model.partial_fit(*next(gen))

score = best_random_model.evaluate(X_test, y_test, verbose=0)[-1]
end_random = time() - start_random

# Random search in this case tends to hold up pretty well given that it only trains one model to completion
# and runs pretty quick (and even then most of its time is being spent compiling the models instead of training them)
# Good idea: train top 5 random models to completion instead of top 1. It would still be faster than grid.

print('Best score from short random search:', score, ', time to complete:', int(end_random), 'seconds, parameters:',
      best_random_model.hyperparameters)
print('Best score from grid search:', best_grid_score, ', time to complete:', int(end_grid), 'seconds, parameters:',
      best_grid_model.hyperparameters)
