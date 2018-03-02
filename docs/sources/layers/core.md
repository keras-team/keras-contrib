<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L731)</span>
### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

Just your regular densely-connected NN layer.

`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).

- __Note__: if the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.

__Example__


```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you don't specify anything, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).

__Input shape__

nD tensor with shape: `(batch_size, ..., input_dim)`.
The most common situation would be
a 2D input with shape `(batch_size, input_dim)`.

__Output shape__

nD tensor with shape: `(batch_size, ..., units)`.
For instance, for a 2D input with shape `(batch_size, input_dim)`,
the output would have shape `(batch_size, units)`.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L271)</span>
### Activation

```python
keras.layers.Activation(activation)
```

Applies an activation function to an output.

__Arguments__

- __activation__: name of activation function to use
(see: [activations](../activations.md)),
or alternatively, a Theano or TensorFlow operation.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L72)</span>
### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

Applies Dropout to the input.

Dropout consists in randomly setting
a fraction `rate` of input units to 0 at each update during training time,
which helps prevent overfitting.

__Arguments__

- __rate__: float between 0 and 1. Fraction of the input units to drop.
- __noise_shape__: 1D integer tensor representing the shape of the
binary dropout mask that will be multiplied with the input.
For instance, if your inputs have shape
`(batch_size, timesteps, features)` and
you want the dropout mask to be the same for all timesteps,
you can use `noise_shape=(batch_size, 1, features)`.
- __seed__: A Python integer to use as random seed.

__References__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L453)</span>
### Flatten

```python
keras.layers.Flatten()
```

Flattens the input. Does not affect the batch size.

__Example__


```python
model = Sequential()
model.add(Conv2D(64, 3, 3,
                 border_mode='same',
                 input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L1375)</span>
### Input

```python
keras.engine.topology.Input()
```

`Input()` is used to instantiate a Keras tensor.

A Keras tensor is a tensor object from the underlying backend
(Theano or TensorFlow), which we augment with certain
attributes that allow us to build a Keras model
just by knowing the inputs and outputs of the model.

For instance, if a, b and c are Keras tensors,
it becomes possible to do:
`model = Model(input=[a, b], output=c)`

The added Keras attributes are:
- __._keras_shape__: Integer shape tuple propagated
via Keras-side shape inference.
- __._keras_history__: Last layer applied to the tensor.
the entire layer graph is retrievable from that layer,
recursively.

__Arguments__

- __shape__: A shape tuple (integer), not including the batch size.
For instance, `shape=(32,)` indicates that the expected input
will be batches of 32-dimensional vectors.
- __batch_shape__: A shape tuple (integer), including the batch size.
For instance, `batch_shape=(10, 32)` indicates that
the expected input will be batches of 10 32-dimensional vectors.
`batch_shape=(None, 32)` indicates batches of an arbitrary number
of 32-dimensional vectors.
- __name__: An optional name string for the layer.
Should be unique in a model (do not reuse the same name twice).
It will be autogenerated if it isn't provided.
- __dtype__: The data type expected by the input, as a string
(`float32`, `float64`, `int32`...)
- __sparse__: A boolean specifying whether the placeholder
to be created is sparse.
- __tensor__: Optional existing tensor to wrap into the `Input` layer.
If set, the layer will not create a placeholder tensor.

__Returns__

A tensor.

__Example__


```python
# this is a logistic regression in Keras
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L302)</span>
### Reshape

```python
keras.layers.Reshape(target_shape)
```

Reshapes an output to a certain shape.

__Arguments__

- __target_shape__: target shape. Tuple of integers.
Does not include the batch axis.

__Input shape__

Arbitrary, although all dimensions in the input shaped must be fixed.
Use the keyword argument `input_shape`
(tuple of integers, does not include the batch axis)
when using this layer as the first layer in a model.

__Output shape__

`(batch_size,) + target_shape`

__Example__


```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L401)</span>
### Permute

```python
keras.layers.Permute(dims)
```

Permutes the dimensions of the input according to a given pattern.

Useful for e.g. connecting RNNs and convnets together.

__Example__


```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

__Arguments__

- __dims__: Tuple of integers. Permutation pattern, does not include the
samples dimension. Indexing starts at 1.
For instance, `(2, 1)` permutes the first and second dimension
of the input.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same as the input shape, but with the dimensions re-ordered according
to the specified pattern.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L488)</span>
### RepeatVector

```python
keras.layers.RepeatVector(n)
```

Repeats the input n times.

__Example__


```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

__Arguments__

- __n__: integer, repetition factor.

__Input shape__

2D tensor of shape `(num_samples, features)`.

__Output shape__

3D tensor of shape `(num_samples, n, features)`.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L530)</span>
### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
```

Wraps arbitrary expression as a `Layer` object.

__Examples__


```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```
```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))
```

__Arguments__

- __function__: The function to be evaluated.
Takes input tensor as first argument.
- __output_shape__: Expected output shape from function.
Only relevant when using Theano.
Can be a tuple or function.
If a tuple, it only specifies the first dimension onward;
sample dimension is assumed either the same as the input:
`output_shape = (input_shape[0], ) + output_shape`
or, the input is `None` and
the sample dimension is also `None`:
`output_shape = (None, ) + output_shape`
If a function, it specifies the entire shape as a function of the
input shape: `output_shape = f(input_shape)`
- __arguments__: optional dictionary of keyword arguments to be passed
to the function.

__Input shape__

Arbitrary. Use the keyword argument input_shape
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Specified by `output_shape` argument
(or auto-inferred when using TensorFlow).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L874)</span>
### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
```

Layer that applies an update to the cost function based input activity.

__Arguments__

- __l1__: L1 regularization factor (positive float).
- __l2__: L2 regularization factor (positive float).

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L25)</span>
### Masking

```python
keras.layers.Masking(mask_value=0.0)
```

Masks a sequence by using a mask value to skip timesteps.

For each timestep in the input tensor (dimension #1 in the tensor),
if all values in the input tensor at that timestep
are equal to `mask_value`, then the timestep will be masked (skipped)
in all downstream layers (as long as they support masking).

If any downstream layer does not support masking yet receives such
an input mask, an exception will be raised.

__Example__


Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
to be fed to an LSTM layer.
You want to mask timestep #3 and #5 because you lack data for
these timesteps. You can:

- set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
- insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```
