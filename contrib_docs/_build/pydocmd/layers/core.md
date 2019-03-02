<h1 id="keras_contrib.layers.CosineDense">CosineDense</h1>

```python
CosineDense(self, units, kernel_initializer='glorot_uniform', activation=None, weights=None, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, **kwargs)
```
A cosine normalized densely-connected NN layer

__Example__


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

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __init__: name of initialization function for the weights of the layer
- __(see [initializers](https__://keras.io/initializers)),
        or alternatively, Theano function to use for weights
        initialization. This parameter is only relevant
        if you don't pass a `weights` argument.
- __activation__: name of activation function to use
- __(see [activations](https__://keras.io/activations)),
        or alternatively, elementwise Python function.
        If you don't specify anything, no activation is applied
- __(ie. "linear" activation__: a(x) = x).
- __weights__: list of Numpy arrays to set as initial weights.
        The list should have 2 elements, of shape `(input_dim, units)`
        and (units,) for weights and biases respectively.
- __kernel_regularizer__: instance of [WeightRegularizer](
- __https__://keras.io/regularizers)
        (eg. L1 or L2 regularization), applied to the main weights matrix.
- __bias_regularizer__: instance of [WeightRegularizer](
- __https__://keras.io/regularizers), applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](
- __https__://keras.io/regularizers), applied to the network output.
- __kernel_constraint__: instance of the [constraints](
- __https__://keras.io/constraints/) module
        (eg. maxnorm, nonneg), applied to the main weights matrix.
- __bias_constraint__: instance of the [constraints](
- __https__://keras.io/constraints/) module, applied to the bias.
- __use_bias__: whether to include a bias
        (i.e. make the layer affine rather than linear).
- __input_dim__: dimensionality of the input (integer). This argument
        (or alternatively, the keyword argument `input_shape`)
        is required when using this layer as the first layer in a model.

__Input shape__

    nD tensor with shape: `(nb_samples, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(nb_samples, input_dim)`.

__Output shape__

    nD tensor with shape: `(nb_samples, ..., units)`.
    For instance, for a 2D input with shape `(nb_samples, input_dim)`,
    the output would have shape `(nb_samples, units)`.

__References__

    - [Cosine Normalization: Using Cosine Similarity Instead
       of Dot Product in Neural Networks](https://arxiv.org/pdf/1702.05870.pdf)

