import numpy as np
import warnings

from keras.callbacks import Callback
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
        # Should work for all layers with relu activation. Tested for Dense and Conv2D
        if 'activation' in layer.get_config():
            return layer.get_config()['activation'] == 'relu'
        else:
            return False

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
                layer_name = self.model.layers[layer_index].name
                # layer_weight is a list [W] (+ [b])
                layer_weight = self.model.layers[layer_index].get_weights()
                # with kernel and bias, the weights are saved as a list [W, b]. If only weights, it is [W]
                assert type(layer_weight) == list
                layer_weight_shape = np.shape(layer_weight[0])
                yield [layer_index, layer_activations, layer_name, layer_weight_shape]

    def on_epoch_end(self, epoch, logs={}):
        for relu_activation in self.get_relu_activations():
            layer_index, activation_values, layer_name, layer_weight_shape = relu_activation

            shape_act = activation_values.shape

            weight_len = len(layer_weight_shape)
            act_len = len(shape_act)

            # should work for both Conv and Flat
            if K.backend() == 'tensorflow':
                # features in last axis
                axis_filter = -1
            elif K.backend() == 'theano':
                # features before the convolution axis, for weight_len the input and output have to be subtracted
                axis_filter = -1 - (weight_len - 2)
            else:
                raise ValueError('Unknown backend: {}'.format(K.backend()))

            total_featuremaps = shape_act[axis_filter]

            axis = tuple(
                i for i in range(act_len) if (i != axis_filter) and (i != (len(shape_act) + axis_filter)))

            dead_neurons = np.sum(np.sum(activation_values, axis=axis) == 0)

            dead_neurons_share = dead_neurons / total_featuremaps
            if (self.verbose and dead_neurons > 0) or dead_neurons_share > self.dead_neurons_share_threshold:
                str_warning = 'Layer {} (#{}) has {} dead neurons ({:.2%})!'.format(layer_name, layer_index,
                                                                                    dead_neurons, dead_neurons_share)
                print(str_warning)
                warnings.warn(str_warning, RuntimeWarning)
