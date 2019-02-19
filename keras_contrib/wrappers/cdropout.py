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

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)), n_data=5000)
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input shape
        model.add(ConcreteDropout(Dense(32), n_data=500))
        # now model.output_shape == (None, 32)

        # Note that the current implementation supports Conv2D Layer as well.
    ```

    # Arguments
        layer: The to be wrapped layer.
        n_data: int. Length of the dataset.
        length_scale: float. Prior lengthscale.
        model_precision: float. Model precision parameter is `1` for classification.
                         Also known as inverse observation noise.
        prob_init: Tuple[float, float].
                   Probability lower / upper bounds of dropout rate initialization.
        temp: float. Temperature.
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
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = length_scale**2 / (model_precision * n_data)
        self.dropout_regularizer = 2 / (model_precision * n_data)
        self.prob_init = tuple(np.log(prob_init))
        self.temp = temp
        self.seed = seed

        self.supports_masking = True
        self.p_logit = None
        self.p = None

    def _concrete_dropout(self, inputs, layer_type):
        """Applies concrete dropout.
           Used at training time (gradients can be propagated).

        # Arguments
            inputs: Input.
            layer_type: str. Either 'dense' or 'conv2d'.
        # Returns
            A tensor with the same shape as inputs and dropout applied.
        """
        assert layer_type in {'dense', 'conv2d'}
        eps = K.cast_to_floatx(K.epsilon())

        noise_shape = K.shape(inputs)
        if layer_type == 'conv2d':
            if K.image_data_format() == 'channels_first':
                noise_shape = (noise_shape[0], noise_shape[1], 1, 1)
            else:
                noise_shape = (noise_shape[0], 1, 1, noise_shape[3])
        unif_noise = K.random_uniform(shape=noise_shape,
                                      seed=self.seed,
                                      dtype=inputs.dtype)
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / self.temp)

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
        elif len(input_shape) == 4:  # Conv2D_layer
            input_dim = (input_shape[1]
                         if K.image_data_format() == 'channels_first'
                         else input_shape[3])  # we drop only channels
        else:
            raise ValueError(
                'concrete_dropout currenty supports only Dense/Conv2D layers')

        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                             shape=(1,),
                                             initializer=RandomUniform(
                                                 *self.prob_init,
                                                 seed=self.seed
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
                'dense' if len(K.int_shape(inputs)) == 2 else 'conv2d'
            )))

        return K.in_train_phase(relaxed_dropped_inputs,
                                self.layer.call(inputs),
                                training=training)

    def get_config(self):
        config = {'weight_regularizer': self.weight_regularizer,
                  'dropout_regularizer': self.dropout_regularizer,
                  'prob_init': tuple(np.round(self.prob_init, 8)),
                  'temp': self.temp,
                  'seed': self.seed}
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
