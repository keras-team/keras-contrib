import numpy as np
import warnings

from keras.callbacks import Callback
from keras.layers import Dense
from keras import backend as K


class DeadReluDetector(Callback):
    """Reports the number of dead ReLUs after each training epoch
    ReLU is considered to be dead if it did not fire once for entire training set

    # Arguments
        x_train: Training dataset to check whether or not neurons fire
        verbose: verbosity mode
            True means that even a single dead neuron triggers warning
            False means that only significant number of dead neurons (10% or more)
            triggers warning
    """
    def __init__(self, x_train, verbose=False):
        super(DeadReluDetector, self).__init__()
        self.x_train = x_train
        self.verbose = verbose
        self.dead_neurons_share_threshold = 0.1

    @staticmethod
    def is_relu_layer(layer):
        return isinstance(layer, Dense) and layer.get_config()['activation'] == 'relu'

    def get_relu_activations(self):
        model_input = self.model.input
        is_multi_input = isinstance(model_input, list)
        if not is_multi_input:
            model_input = [model_input]

        funcs = [K.function(model_input + [K.learning_phase()], [layer.output]) for layer in self.model.layers]
        if is_multi_input:
            list_inputs = []
            list_inputs.extend(self.x_train)
            list_inputs.append(1.)
        else:
            list_inputs = [self.x_train, 1.]

        layer_outputs = [func(list_inputs)[0] for func in funcs]
        for layer_index, layer_activations in enumerate(layer_outputs):
            if self.is_relu_layer(self.model.layers[layer_index]):
                yield [layer_index, layer_activations]

    def on_epoch_end(self, epoch, logs={}):
        for relu_activation in self.get_relu_activations():
            layer_index, activation_values = relu_activation
            total_neurons = activation_values.shape[-1]
            dead_neurons = np.sum(activation_values == 0)
            dead_neurons_share = dead_neurons / total_neurons
            if (self.verbose and dead_neurons > 0) or dead_neurons_share > self.dead_neurons_share_threshold:
                warnings.warn(
                    'Layer #{} has {} dead neurons ({:.2%})!'
                        .format(layer_index, dead_neurons, dead_neurons_share),
                    RuntimeWarning
                )
