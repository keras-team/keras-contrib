<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/advanced_activations.py#L11)</span>
### PELU

```python
keras_contrib.layers.PELU(alpha_initializer='ones', alpha_regularizer=None, alpha_constraint=None, beta_initializer='ones', beta_regularizer=None, beta_constraint=None, shared_axes=None)
```

Parametric Exponential Linear Unit.
It follows:
`f(x) = alphas * (exp(x / betas) - 1) for x < 0`,
`f(x) = (alphas / betas) * x for x >= 0`,
where `alphas` & `betas` are learned arrays with the same shape as x.
__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.
__Output shape__

Same shape as the input.
__Arguments__

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
__References__

- [PARAMETRIC EXPONENTIAL LINEAR UNIT FOR DEEP CONVOLUTIONAL NEURAL NETWORKS](https://arxiv.org/abs/1605.09332v3)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/advanced_activations.py#L119)</span>
### SReLU

```python
keras_contrib.layers.initializers.RandomUniform object at 0x7f9ba5c71ef0>, t_right_initializer=<keras.initializers.RandomUniform object at 0x7f9ba5c71f28>, a_right_initializer='ones', shared_axes=None)
```

S-shaped Rectified Linear Unit.

It follows:
`f(x) = t^r + a^r(x - t^r) for x >= t^r`,
`f(x) = x for t^r > x > t^l`,
`f(x) = t^l + a^l(x - t^l) for x <= t^l`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __t_left_initializer__: initializer function for the left part intercept
- __a_left_initializer__: initializer function for the left part slope
- __t_right_initializer__: initializer function for the right part intercept
- __a_right_initializer__: initializer function for the right part slope
- __shared_axes__: the axes along which to share learnable
    parameters for the activation function.
    For example, if the incoming feature maps
    are from a 2D convolution
    with output shape `(batch, height, width, channels)`,
    and you wish to share parameters across space
    so that each filter only has one set of parameters,
    set `shared_axes=[1, 2]`.

__References__

- [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/advanced_activations.py#L242)</span>
### Swish

```python
keras_contrib.layers.Swish(beta=1.0, trainable=False)
```

Swish (Ramachandranet al., 2017)

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __beta__: float >= 0. Scaling factor
    if set to 1 and trainable set to False (default), Swish equals the SiLU activation (Elfwing et al., 2017)
- __trainable__: whether to learn the scaling factor during training or not

__References__

- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
- [Sigmoid-weighted linear units for neural network function approximation in reinforcement learning](https://arxiv.org/abs/1702.03118)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/advanced_activations.py#L289)</span>
### SineReLU

```python
keras_contrib.layers.SineReLU(epsilon=0.0025)
```

Sine Rectified Linear Unit to generate oscilations.

It allows an oscilation in the gradients when the weights are negative.
The oscilation can be controlled with a parameter, which makes it be close
or equal to zero. The functional is diferentiable at any point due to its derivative.
For instance, at 0, the derivative of 'sin(0) - cos(0)' is 'cos(0) + sin(0)' which is 1.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __epsilon__: float. Hyper-parameter used to control the amplitude of the sinusoidal wave when weights are negative.
         The default value, 0.0025, since it works better for CNN layers and those are the most used layers nowadays.
         When using Dense Networks, try something around 0.006.

__References:__

- SineReLU: An Alternative to the ReLU Activation Function. This function was

first introduced at the Codemotion Amsterdam 2018 and then at the DevDays, in Vilnius, Lithuania.
It has been extensively tested with Deep Nets, CNNs, LSTMs, Residual Nets and GANs, based
on the MNIST, Kaggle Toxicity and IMDB datasets.

__Performance:__


- Fashion MNIST
  * Mean of 6 runs per Activation Function
    * Fully Connection Network
      - SineReLU: loss mean -> 0.3522; accuracy mean -> 89.18; mean of std loss -> 0.08375204467435822
      - LeakyReLU: loss mean-> 0.3553; accuracy mean -> 88.98; mean of std loss -> 0.0831161868455245
      - ReLU: loss mean -> 0.3519; accuracy mean -> 88.84; mean of std loss -> 0.08358816501301362

    * Convolutional Neural Network

      - SineReLU: loss mean -> 0.2180; accuracy mean -> 92.49; mean of std loss -> 0.0781155784858847
      - LeakyReLU: loss mean -> 0.2205; accuracy mean -> 92.37; mean of std loss -> 0.09273670474788205
      - ReLU: loss mean -> 0.2144; accuracy mean -> 92.45; mean of std loss -> 0.09396114585977
- MNIST
  * Mean of 6 runs per Activation Function
    * Fully Connection Network
      - SineReLU: loss mean -> 0.0623; accuracy mean -> 98.53; mean of std loss -> 0.06012015231824904
      - LeakyReLU: loss mean-> 0.0623; accuracy mean -> 98.50; mean of std loss -> 0.06052147632835356
      - ReLU: loss mean -> 0.0605; accuracy mean -> 98.49; mean of std loss -> 0.059599885665016096

    * Convolutional Neural Network

      - SineReLU: loss mean -> 0.0198; accuracy mean -> 99.51; mean of std loss -> 0.0425338329550847
      - LeakyReLU: loss mean -> 0.0216; accuracy mean -> 99.40; mean of std loss -> 0.04834468835196667
      - ReLU: loss mean -> 0.0185; accuracy mean -> 99.49; mean of std loss -> 0.05503719489690131

__Jupyter Notebooks__

- https://github.com/ekholabs/DLinK/blob/master/notebooks/keras

__Examples__

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
