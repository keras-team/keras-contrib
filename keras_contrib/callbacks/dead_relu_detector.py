import numpy as np

from keras.callbacks import Callback
from keras import backend as K


class DeadReluDetector(Callback):
    """Reports the number of dead ReLUs after each training epoch
    ReLU is considered to be dead if it did not fire once for entire training set

    # Arguments
        x_train: Training dataset to check whether or not neurons fire
        verbose: verbosity mode
            True means that even a single dead neuron triggers a warning message
            False means that only significant number of dead neurons (10% or more)
            triggers a warning message
    """

    def __init__(self, x_train, verbose=False):
        super(DeadReluDetector, self).__init__()
        self.x_train = x_train
        self.verbose = verbose
        self.dead_neurons_share_threshold = 0.1

    @staticmethod
    def is_relu_layer(layer):
        # Should work for all layers with relu
        # activation. Tested for Dense and Conv2D
        return layer.get_config().get('activation', None) == 'relu'

    def get_relu_activations(self):
        model_input = self.model.input
        is_multi_input = isinstance(model_input, list)
        if not is_multi_input:
            model_input = [model_input]

        funcs = {}
        for index, layer in enumerate(self.model.layers):
            if not layer.get_weights():
                continue
            funcs[index] = K.function(model_input
                                      + [K.learning_phase()], [layer.output])

        if is_multi_input:
            list_inputs = []
            list_inputs.extend(self.x_train)
            list_inputs.append(1.)
        else:
            list_inputs = [self.x_train, 1.]

        layer_outputs = {}
        for index, func in funcs.items():
            layer_outputs[index] = func(list_inputs)[0]

        for layer_index, layer_activations in layer_outputs.items():
            if self.is_relu_layer(self.model.layers[layer_index]):
                layer_name = self.model.layers[layer_index].name
                # layer_weight is a list [W] (+ [b])
                layer_weight = self.model.layers[layer_index].get_weights()

                # with kernel and bias, the weights are saved as a list [W, b].
                # If only weights, it is [W]
                if type(layer_weight) is not list:
                    raise ValueError("'Layer_weight' should be a list, "
                                     "but was {}".format(type(layer_weight)))

                # there are no weights for current layer; skip it
                # this is only legitimate if layer is "Activation"
                if len(layer_weight) == 0:
                    continue

                layer_weight_shape = np.shape(layer_weight[0])
                yield [layer_index,
                       layer_activations,
                       layer_name,
                       layer_weight_shape]

    def on_epoch_end(self, epoch, logs={}):
        for relu_activation in self.get_relu_activations():
            layer_index = relu_activation[0]
            activation_values = relu_activation[1]
            layer_name = relu_activation[2]
            layer_weight_shape = relu_activation[3]

            shape_act = activation_values.shape

            weight_len = len(layer_weight_shape)
            act_len = len(shape_act)

            # should work for both Conv and Flat
            if K.image_data_format() == 'channels_last':
                # features in last axis
                axis_filter = -1
            else:
                # features before the convolution axis, for weight_
                # len the input and output have to be subtracted
                axis_filter = -1 - (weight_len - 2)

            total_featuremaps = shape_act[axis_filter]

            axis = []
            for i in range(act_len):
                if (i != axis_filter) and (i != (len(shape_act) + axis_filter)):
                    axis.append(i)
            axis = tuple(axis)

            dead_neurons = np.sum(np.sum(activation_values, axis=axis) == 0)

            dead_neurons_share = float(dead_neurons) / float(total_featuremaps)
            if ((self.verbose and dead_neurons > 0)
                    or dead_neurons_share >= self.dead_neurons_share_threshold):
                str_warning = ('Layer {} (#{}) has {} '
                               'dead neurons ({:.2%})!').format(layer_name,
                                                                layer_index,
                                                                dead_neurons,
                                                                dead_neurons_share)
                print(str_warning)
