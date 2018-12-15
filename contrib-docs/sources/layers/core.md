<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/core.py#L15)</span>
### CosineDense

```python
keras_contrib.layers.CosineDense(units, kernel_initializer='glorot_uniform', activation=None, weights=None, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, input_dim=None)
```

A cosine normalized densely-connected NN layer
Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks
https://arxiv.org/pdf/1702.05870.pdf

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

**Note that a regular Dense layer may work better as the final layer
```

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __init__: name of initialization function for the weights of the layer
    (see [initializers](../initializers.md)),
    or alternatively, Theano function to use for weights
    initialization. This parameter is only relevant
    if you don't pass a `weights` argument.
- __activation__: name of activation function to use
    (see [activations](../activations.md)),
    or alternatively, elementwise Theano function.
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: a(x) = x).
- __weights__: list of Numpy arrays to set as initial weights.
    The list should have 2 elements, of shape `(input_dim, units)`
    and (units,) for weights and biases respectively.
- __kernel_regularizer__: instance of [WeightRegularizer](../regularizers.md)
    (eg. L1 or L2 regularization), applied to the main weights matrix.
- __bias_regularizer__: instance of [WeightRegularizer](../regularizers.md),
    applied to the bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
    applied to the network output.
- __kernel_constraint__: instance of the [constraints](../constraints.md) module
    (eg. maxnorm, nonneg), applied to the main weights matrix.
- __bias_constraint__: instance of the [constraints](../constraints.md) module,
    applied to the bias.
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
