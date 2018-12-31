from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras.layers import InputSpec
from keras import backend as K
from keras.utils import get_custom_objects


class PELU(Layer):
    """Parametric Exponential Linear Unit.

    It follows:
    `f(x) = alphas * (exp(x / betas) - 1) for x < 0`,
    `f(x) = (alphas / betas) * x for x >= 0`,
    where `alphas` & `betas` are learned arrays with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alphas_initializer: initialization function for the alpha variable weights.
        betas_initializer: initialization function for the beta variable weights.
        weights: initial weights, as a list of a single Numpy array.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    # References
        - [Parametric exponential linear unit for deep convolutional neural networks](
           https://arxiv.org/abs/1605.09332v3)
    """

    def __init__(self, alpha_initializer='ones',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 beta_initializer='ones',
                 beta_regularizer=None,
                 beta_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        param_shape = tuple(param_shape)
        # Initialised as ones to emulate the default ELU
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.beta = self.add_weight(shape=param_shape,
                                    name='beta',
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)

        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            pos = K.relu(x) * (K.pattern_broadcast(self.alpha, self.param_broadcast) /
                               K.pattern_broadcast(self.beta, self.param_broadcast))
            neg = (K.pattern_broadcast(self.alpha, self.param_broadcast) *
                   (K.exp((-K.relu(-x))
                          / K.pattern_broadcast(self.beta, self.param_broadcast)) - 1))
        else:
            pos = K.relu(x) * self.alpha / self.beta
            neg = self.alpha * (K.exp((-K.relu(-x)) / self.beta) - 1)
        return neg + pos

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(PELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'PELU': PELU})


class SReLU(Layer):
    """S-shaped Rectified Linear Unit.

    It follows:
    `f(x) = t^r + a^r(x - t^r) for x >= t^r`,
    `f(x) = x for t^r > x > t^l`,
    `f(x) = t^l + a^l(x - t^l) for x <= t^l`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        t_left_initializer: initializer function for the left part intercept
        a_left_initializer: initializer function for the left part slope
        t_right_initializer: initializer function for the right part intercept
        a_right_initializer: initializer function for the right part slope
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    # References
        - [Deep Learning with S-shaped Rectified Linear Activation Units](
           http://arxiv.org/abs/1512.07030)
    """

    def __init__(self, t_left_initializer='zeros',
                 a_left_initializer=initializers.RandomUniform(minval=0, maxval=1),
                 t_right_initializer=initializers.RandomUniform(minval=0, maxval=5),
                 a_right_initializer='ones',
                 shared_axes=None,
                 **kwargs):
        super(SReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.t_left_initializer = initializers.get(t_left_initializer)
        self.a_left_initializer = initializers.get(a_left_initializer)
        self.t_right_initializer = initializers.get(t_right_initializer)
        self.a_right_initializer = initializers.get(a_right_initializer)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        param_shape = tuple(param_shape)

        self.t_left = self.add_weight(shape=param_shape,
                                      name='t_left',
                                      initializer=self.t_left_initializer)

        self.a_left = self.add_weight(shape=param_shape,
                                      name='a_left',
                                      initializer=self.a_left_initializer)

        self.t_right = self.add_weight(shape=param_shape,
                                       name='t_right',
                                       initializer=self.t_right_initializer)

        self.a_right = self.add_weight(shape=param_shape,
                                       name='a_right',
                                       initializer=self.a_right_initializer)

        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, x, mask=None):
        # ensure the the right part is always to the right of the left
        t_right_actual = self.t_left + K.abs(self.t_right)

        if K.backend() == 'theano':
            t_left = K.pattern_broadcast(self.t_left, self.param_broadcast)
            a_left = K.pattern_broadcast(self.a_left, self.param_broadcast)
            a_right = K.pattern_broadcast(self.a_right, self.param_broadcast)
            t_right_actual = K.pattern_broadcast(t_right_actual,
                                                 self.param_broadcast)
        else:
            t_left = self.t_left
            a_left = self.a_left
            a_right = self.a_right

        y_left_and_center = t_left + K.relu(x - t_left,
                                            a_left,
                                            t_right_actual - t_left)
        y_right = K.relu(x - t_right_actual) * a_right
        return y_left_and_center + y_right

    def get_config(self):
        config = {
            't_left_initializer': self.t_left_initializer,
            'a_left_initializer': self.a_left_initializer,
            't_right_initializer': self.t_right_initializer,
            'a_right_initializer': self.a_right_initializer,
            'shared_axes': self.shared_axes
        }
        base_config = super(SReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'SReLU': SReLU})


class Swish(Layer):
    """ Swish (Ramachandranet al., 2017)

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        beta: float >= 0. Scaling factor
            if set to 1 and trainable set to False (default),
            Swish equals the SiLU activation (Elfwing et al., 2017)
        trainable: whether to learn the scaling factor during training or not

    # References
        - [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        - [Sigmoid-weighted linear units for neural network function
           approximation in reinforcement learning](https://arxiv.org/abs/1702.03118)
    """

    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.scaling_factor = K.variable(self.beta,
                                         dtype=K.floatx(),
                                         name='scaling_factor')
        if self.trainable:
            self._trainable_weights.append(self.scaling_factor)
        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return inputs * K.sigmoid(self.scaling_factor * inputs)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'Swish': Swish})


class SineReLU(Layer):
    """Sine Rectified Linear Unit to generate oscilations.

    It allows an oscilation in the gradients when the weights are negative.
    The oscilation can be controlled with a parameter, which makes it be close
    or equal to zero. The functional is diferentiable at any point due to
    its derivative.
    For instance, at 0, the derivative of 'sin(0) - cos(0)'
    is 'cos(0) + sin(0)' which is 1.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        epsilon: float. Hyper-parameter used to control the amplitude of the
            sinusoidal wave when weights are negative.
            The default value, 0.0025, since it works better for CNN layers and
            those are the most used layers nowadays.
            When using Dense Networks, try something around 0.006.

    # References:
        - [SineReLU: An Alternative to the ReLU Activation Function](
           https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d).

        This function was
        first introduced at the Codemotion Amsterdam 2018 and then at
        the DevDays, in Vilnius, Lithuania.
        It has been extensively tested with Deep Nets, CNNs,
        LSTMs, Residual Nets and GANs, based
        on the MNIST, Kaggle Toxicity and IMDB datasets.

    # Performance:

        - Fashion MNIST
          * Mean of 6 runs per Activation Function
            * Fully Connection Network
              - SineReLU: loss mean -> 0.3522; accuracy mean -> 89.18;
                  mean of std loss -> 0.08375204467435822
              - LeakyReLU: loss mean-> 0.3553; accuracy mean -> 88.98;
              mean of std loss -> 0.0831161868455245
              - ReLU: loss mean -> 0.3519; accuracy mean -> 88.84;
              mean of std loss -> 0.08358816501301362
            * Convolutional Neural Network
              - SineReLU: loss mean -> 0.2180; accuracy mean -> 92.49;
              mean of std loss -> 0.0781155784858847
              - LeakyReLU: loss mean -> 0.2205; accuracy mean -> 92.37;
              mean of std loss -> 0.09273670474788205
              - ReLU: loss mean -> 0.2144; accuracy mean -> 92.45;
              mean of std loss -> 0.09396114585977
        - MNIST
          * Mean of 6 runs per Activation Function
            * Fully Connection Network
              - SineReLU: loss mean -> 0.0623; accuracy mean -> 98.53;
              mean of std loss -> 0.06012015231824904
              - LeakyReLU: loss mean-> 0.0623; accuracy mean -> 98.50;
              mean of std loss -> 0.06052147632835356
              - ReLU: loss mean -> 0.0605; accuracy mean -> 98.49;
              mean of std loss -> 0.059599885665016096
            * Convolutional Neural Network
              - SineReLU: loss mean -> 0.0198; accuracy mean -> 99.51;
              mean of std loss -> 0.0425338329550847
              - LeakyReLU: loss mean -> 0.0216; accuracy mean -> 99.40;
              mean of std loss -> 0.04834468835196667
              - ReLU: loss mean -> 0.0185; accuracy mean -> 99.49;
              mean of std loss -> 0.05503719489690131

    # Jupyter Notebooks
        - https://github.com/ekholabs/DLinK/blob/master/notebooks/keras

    # Examples
        The Advanced Activation function SineReLU have to be imported from the
        keras_contrib.layers package.

        To see full source-code of this architecture and other examples,
        please follow this link: https://github.com/ekholabs/DLinK

        ```python
            model = Sequential()
            model.add(Dense(128, input_shape = (784,)))
            model.add(SineReLU())
            model.add(Dropout(0.2))

            model.add(Dense(256))
            model.add(SineReLU())
            model.add(Dropout(0.3))

            model.add(Dense(1024))
            model.add(SineReLU())
            model.add(Dropout(0.5))

            model.add(Dense(10, activation = 'softmax'))
        ```
    """

    def __init__(self, epsilon=0.0025, **kwargs):
        super(SineReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = K.cast_to_floatx(epsilon)

    def call(self, Z):
        m = self.epsilon * (K.sin(Z) - K.cos(Z))
        A = K.maximum(m, Z)
        return A

    def get_config(self):
        config = {'epsilon': float(self.epsilon)}
        base_config = super(SineReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

get_custom_objects().update({'SineReLU': SineReLU})
