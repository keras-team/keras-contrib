<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L11)</span>
### GaussianNoise

```python
keras.layers.GaussianNoise(stddev)
```

Apply additive zero-centered Gaussian noise.

This is useful to mitigate overfitting
(you could see it as a form of random data augmentation).
Gaussian Noise (GS) is a natural choice as corruption process
for real valued inputs.

As it is a regularization layer, it is only active at training time.

__Arguments__

- __stddev__: float, standard deviation of the noise distribution.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L52)</span>
### GaussianDropout

```python
keras.layers.GaussianDropout(rate)
```

Apply multiplicative 1-centered Gaussian noise.

As it is a regularization layer, it is only active at training time.

__Arguments__

- __rate__: float, drop probability (as with `Dropout`).
The multiplicative noise will have
standard deviation `sqrt(rate / (1 - rate))`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L96)</span>
### AlphaDropout

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```

Applies Alpha Dropout to the input.

Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
to their original values, in order to ensure the self-normalizing property
even after this dropout.
Alpha Dropout fits well to Scaled Exponential Linear Units
by randomly setting activations to the negative saturation value.

__Arguments__

- __rate__: float, drop probability (as with `Dropout`).
The multiplicative noise will have
standard deviation `sqrt(rate / (1 - rate))`.
- __seed__: A Python integer to use as random seed.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
