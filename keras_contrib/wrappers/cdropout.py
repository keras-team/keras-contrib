# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from keras import backend as K
from keras.initializers import RandomUniform
from keras.layers import InputSpec
from keras.layers.wrappers import Wrapper
from keras_contrib.utils.test_utils import to_tuple


class ConcreteDropout(Wrapper):
    """A wrapper automating the dropout rate choice
       through the 'Concrete Dropout' technique.

       Note that currently only Dense layers with weights
       and Conv layers (Conv1D, Conv2D, Conv3D) are supported.
       In the case of Dense Layers, dropout is applied to its complete input,
       whereas in the Conv case just the input-channels are dropped.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)), n_data=5000)
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input shape
        model.add(ConcreteDropout(Dense(32), n_data=500))
        # now model.output_shape == (None, 32)

        # Note that the current implementation supports Conv layers as well.
    ```

    # Arguments
        layer: The to be wrapped layer.
        n_data: int. `n_data > 0`.
                Length of the dataset.
        length_scale: float. `length_scale > 0`.
                      Prior lengthscale.
        model_precision: float. `model_precision > 0`.
                         Model precision parameter is `1` for classification.
                         Also known as inverse observation noise.
        prob_init: Tuple[float, float]. `prob_init > 0`
                   Probability lower / upper bounds of dropout rate initialization.
        temp: float. Temperature. `temp > 0`.
              Determines the speed of probability (i.e. dropout rate) adjustments.
        seed: Seed for random probability sampling.

    # References
        - [Concrete Dropout](https://arxiv.org/pdf/1705.07832.pdf)
    """

    def __init__(self,
                 layer,
                 n_data,
                 length_scale=5e-2,
                 model_precision=1,
                 prob_init=(0.1, 0.5),
                 temp=0.4,
                 seed=None,
                 **kwargs):
        assert 'kernel_regularizer' not in kwargs
        assert n_data > 0 and isinstance(n_data, int)
        assert length_scale > 0.
        assert prob_init[0] <= prob_init[1] and prob_init[0] > 0.
        assert temp > 0.
        assert model_precision > 0.
        super(ConcreteDropout, self).__init__(layer, **kwargs)

        self._n_data = n_data
        self._length_scale = length_scale
        self._model_precision = model_precision
        self._prob_init = prob_init
        self._temp = temp
        self._seed = seed

        eps = K.epsilon()
        self.weight_regularizer = length_scale**2 / (model_precision * n_data + eps)
        self.dropout_regularizer = 2 / (model_precision * n_data + eps)
        self.supports_masking = True
        self.p_logit = None
        self.p = None

    def _concrete_dropout(self, inputs, layer_type):
        """Applies concrete dropout.
           Used at training time (gradients can be propagated).

        # Arguments
            inputs: Input.
            layer_type: str. Either 'dense' or 'conv'.
        # Returns
            A tensor with the same shape as inputs and dropout applied.
        """
        assert layer_type in {'dense', 'conv'}
        eps = K.cast_to_floatx(K.epsilon())

        noise_shape = K.shape(inputs)
        if layer_type == 'conv':
            nodrops = np.ones(len(K.int_shape(inputs)) - 2, int)
            _ = lambda *x: x  # don't ask... py2 can't unpack directly into a tuple
            if K.image_data_format() == 'channels_first':
                noise_shape = _(noise_shape[0], noise_shape[1], *nodrops)
            else:
                noise_shape = _(noise_shape[0], *(_(*nodrops) + (noise_shape[-1],)))
        unif_noise = K.random_uniform(shape=noise_shape,
                                      seed=self._seed,
                                      dtype=inputs.dtype)
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / self._temp)

        # apply dropout
        random_tensor = 1. - drop_prob
        retain_prob = 1. - self.p
        inputs *= random_tensor
        inputs /= retain_prob

        return inputs

    def build(self, input_shape=None):
        input_shape = to_tuple(input_shape)
        if len(input_shape) == 2:  # Dense_layer
            input_dim = np.prod(input_shape[-1])  # we drop only last dim
        elif 3 <= len(input_shape) <= 5:  # Conv_layers
            input_dim = (
                input_shape[1]
                if K.image_data_format() == 'channels_first'
                else input_shape[-1]  # we drop only channels
            )
        else:
            raise ValueError(
                'concrete_dropout currenty supports only Dense/Conv layers')

        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                             shape=(1,),
                                             initializer=RandomUniform(
                                                 *np.log(self._prob_init),
                                                 seed=self._seed
                                             ),
                                             trainable=True)
        self.p = K.squeeze(K.sigmoid(self.p_logit), axis=0)

        super(ConcreteDropout, self).build(input_shape)

        # initialize regularizer / prior KL term and add to layer-loss
        weight = self.layer.kernel
        kernel_regularizer = (
            self.weight_regularizer
            * K.sum(K.square(weight))
            / (1. - self.p)
        )
        dropout_regularizer = (
            self.p * K.log(self.p)
            + (1. - self.p) * K.log(1. - self.p)
        ) * self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def call(self, inputs, training=None):
        def relaxed_dropped_inputs():
            return self.layer.call(self._concrete_dropout(inputs, (
                'dense' if len(K.int_shape(inputs)) == 2 else 'conv'
            )))

        return K.in_train_phase(relaxed_dropped_inputs,
                                self.layer.call(inputs),
                                training=training)

    def get_config(self):
        config = {'n_data': self._n_data,
                  'length_scale': self._length_scale,
                  'model_precision': self._model_precision,
                  'prob_init': self._prob_init,
                  'temp': self._temp,
                  'seed': self._seed}
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
