# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import InputSpec
from keras.layers import Layer
from keras_contrib.utils.test_utils import to_tuple


class CosineDense(Layer):
    """A cosine normalized densely-connected NN layer

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(CosineDense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(CosineDense(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(CosineDense(32))

        # Note that a regular Dense layer may work better as the final layer
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        init: name of initialization function for the weights of the layer
            (see [initializers](https://keras.io/initializers)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](https://keras.io/activations)),
            or alternatively, elementwise Python function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, units)`
            and (units,) for weights and biases respectively.
        kernel_regularizer: instance of [WeightRegularizer](
            https://keras.io/regularizers)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        bias_regularizer: instance of [WeightRegularizer](
            https://keras.io/regularizers), applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](
            https://keras.io/regularizers), applied to the network output.
        kernel_constraint: instance of the [constraints](
            https://keras.io/constraints/) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        bias_constraint: instance of the [constraints](
            https://keras.io/constraints/) module, applied to the bias.
        use_bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., units)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, units)`.

    # References
        - [Cosine Normalization: Using Cosine Similarity Instead
           of Dot Product in Neural Networks](https://arxiv.org/pdf/1702.05870.pdf)
    """

    def __init__(self, units, kernel_initializer='glorot_uniform',
                 activation=None, weights=None,
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.units = units

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.initial_weights = weights
        super(CosineDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = to_tuple(input_shape)
        ndim = len(input_shape)
        assert ndim >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim=ndim)]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='zero',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        if self.use_bias:
            b, xb = self.bias, 1.
        else:
            b, xb = 0., 0.

        xnorm = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)
                       + xb
                       + K.epsilon())
        Wnorm = K.sqrt(K.sum(K.square(self.kernel), axis=0)
                       + K.square(b)
                       + K.epsilon())

        xWnorm = (xnorm * Wnorm)

        output = K.dot(x, self.kernel) / xWnorm
        if self.use_bias:
            output += (self.bias / xWnorm)
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        assert input_shape
        assert len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'use_bias': self.use_bias
        }
        base_config = super(CosineDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
