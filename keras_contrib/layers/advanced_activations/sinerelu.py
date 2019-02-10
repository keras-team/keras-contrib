import keras.backend as K
from keras.layers import Layer


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
