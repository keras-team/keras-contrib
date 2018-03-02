<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L196)</span>
### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

Base class for recurrent layers.

__Arguments__

- __cell__: A RNN cell instance. A RNN cell is a class that has:
- a `call(input_at_t, states_at_t)` method, returning
`(output_at_t, states_at_t_plus_1)`. The call method of the
cell can also take the optional argument `constants`, see
section "Note on passing external constants" below.
- a `state_size` attribute. This can be a single integer
(single state) in which case it is
the size of the recurrent state
(which should be the same as the size of the cell output).
This can also be a list/tuple of integers
(one size per state). In this case, the first entry
(`state_size[0]`) should be the same as
the size of the cell output.
It is also possible for `cell` to be a list of RNN cell instances,
in which cases the cells get stacked on after the other in the RNN,
implementing an efficient stacked RNN.
- __return_sequences__: Boolean. Whether to return the last output.
in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
in addition to the output.
- __go_backwards__: Boolean (default False).
If True, process the input sequence backwards and return the
reversed sequence.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
- __unroll__: Boolean (default False).
If True, the network will be unrolled,
else a symbolic loop will be used.
Unrolling can speed-up a RNN,
although it tends to be more memory-intensive.
Unrolling is only suitable for short sequences.
- __input_dim__: dimensionality of the input (integer).
This argument (or alternatively,
the keyword argument `input_shape`)
is required when using this layer as the first layer in a model.
- __input_length__: Length of input sequences, to be specified
when it is constant.
This argument is required if you are going to connect
`Flatten` then `Dense` layers upstream
(without it, the shape of the dense outputs cannot be computed).
Note that if the recurrent layer is not the first layer
in your model, you would need to specify the input length
at the level of the first layer
(e.g. via the `input_shape` argument)

__Input shape__

3D tensor with shape `(batch_size, timesteps, input_dim)`.

__Output shape__

- if `return_state`: a list of tensors. The first tensor is
the output. The remaining tensors are the last states,
each with shape `(batch_size, units)`.
- if `return_sequences`: 3D tensor with shape
`(batch_size, timesteps, units)`.
- else, 2D tensor with shape `(batch_size, units)`.

__Masking__

This layer supports masking for input data with a variable number
of timesteps. To introduce masks to your data,
use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
set to `True`.

__Note on using statefulness in RNNs__

You can set RNN layers to be 'stateful', which means that the states
computed for the samples in one batch will be reused as initial states
for the samples in the next batch. This assumes a one-to-one mapping
between samples in different successive batches.

To enable statefulness:
- specify `stateful=True` in the layer constructor.
- specify a fixed batch size for your model, by passing
if sequential model:
`batch_input_shape=(...)` to the first layer in your model.
else for functional model with 1 or more Input layers:
`batch_shape=(...)` to all the first layers in your model.
This is the expected shape of your inputs
*including the batch size*.
It should be a tuple of integers, e.g. `(32, 10, 100)`.
- specify `shuffle=False` when calling fit().

To reset the states of your model, call `.reset_states()` on either
a specific layer, or on your entire model.

__Note on specifying the initial state of RNNs__

You can specify the initial state of RNN layers symbolically by
calling them with the keyword argument `initial_state`. The value of
`initial_state` should be a tensor or list of tensors representing
the initial state of the RNN layer.

You can specify the initial state of RNN layers numerically by
calling `reset_states` with the keyword argument `states`. The value of
`states` should be a numpy array or list of numpy arrays representing
the initial state of the RNN layer.

__Note on passing external constants to RNNs__

You can pass "external" constants to the cell using the `constants`
keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
requires that the `cell.call` method accepts the same keyword argument
`constants`. Such constants can be used to condition the cell
transformation on additional static inputs (not changing over time),
a.k.a. an attention mechanism.

__Examples__


```python
# First, let's define a RNN Cell, as a layer subclass.

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L918)</span>
### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

Fully-connected RNN where the output is to be fed back to input.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.
- __return_sequences__: Boolean. Whether to return the last output.
in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
in addition to the output.
- __go_backwards__: Boolean (default False).
If True, process the input sequence backwards and return the
reversed sequence.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
- __unroll__: Boolean (default False).
If True, the network will be unrolled,
else a symbolic loop will be used.
Unrolling can speed-up a RNN,
although it tends to be more memory-intensive.
Unrolling is only suitable for short sequences.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1369)</span>
### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

Gated Recurrent Unit - Cho et al. 2014.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __recurrent_activation__: Activation function to use
for the recurrent step
(see [activations](../activations.md)).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.
- __implementation__: Implementation mode, either 1 or 2.
Mode 1 will structure its operations as a larger number of
smaller dot products and additions, whereas mode 2 will
batch them into fewer, larger operations. These modes will
have different performance profiles on different hardware and
for different applications.
- __return_sequences__: Boolean. Whether to return the last output.
in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
in addition to the output.
- __go_backwards__: Boolean (default False).
If True, process the input sequence backwards and return the
reversed sequence.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
- __unroll__: Boolean (default False).
If True, the network will be unrolled,
else a symbolic loop will be used.
Unrolling can speed-up a RNN,
although it tends to be more memory-intensive.
Unrolling is only suitable for short sequences.

__References__

- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1870)</span>
### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

Long-Short Term Memory layer - Hochreiter 1997.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __recurrent_activation__: Activation function to use
for the recurrent step
(see [activations](../activations.md)).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __unit_forget_bias__: Boolean.
If True, add 1 to the bias of the forget gate at initialization.
Setting it to true will also force `bias_initializer="zeros"`.
This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.
- __implementation__: Implementation mode, either 1 or 2.
Mode 1 will structure its operations as a larger number of
smaller dot products and additions, whereas mode 2 will
batch them into fewer, larger operations. These modes will
have different performance profiles on different hardware and
for different applications.
- __return_sequences__: Boolean. Whether to return the last output.
in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
in addition to the output.
- __go_backwards__: Boolean (default False).
If True, process the input sequence backwards and return the
reversed sequence.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
- __unroll__: Boolean (default False).
If True, the network will be unrolled,
else a symbolic loop will be used.
Unrolling can speed-up a RNN,
although it tends to be more memory-intensive.
Unrolling is only suitable for short sequences.

__References__

- [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L165)</span>
### ConvLSTM2D

```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

Convolutional LSTM.

It is similar to an LSTM layer, but the input transformations
and recurrent transformations are both convolutional.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
(i.e. the number output of filters in the convolution).
- __kernel_size__: An integer or tuple/list of n integers, specifying the
dimensions of the convolution window.
- __strides__: An integer or tuple/list of n integers,
specifying the strides of the convolution.
Specifying any stride value != 1 is incompatible with specifying
any `dilation_rate` value != 1.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
one of `channels_last` (default) or `channels_first`.
The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch, time, ..., channels)`
while `channels_first` corresponds to
inputs with shape `(batch, time, channels, ...)`.
It defaults to the `image_data_format` value found in your
Keras config file at `~/.keras/keras.json`.
If you never set it, then it will be "channels_last".
- __dilation_rate__: An integer or tuple/list of n integers, specifying
the dilation rate to use for dilated convolution.
Currently, specifying any `dilation_rate` value != 1 is
incompatible with specifying any `strides` value != 1.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you don't specify anything, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __recurrent_activation__: Activation function to use
for the recurrent step
(see [activations](../activations.md)).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __unit_forget_bias__: Boolean.
If True, add 1 to the bias of the forget gate at initialization.
Use in combination with `bias_initializer="zeros"`.
This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __return_sequences__: Boolean. Whether to return the last output
in the output sequence, or the full sequence.
- __go_backwards__: Boolean (default False).
If True, rocess the input sequence backwards.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.

__Input shape__

- if data_format='channels_first'
5D tensor with shape:
`(samples,time, channels, rows, cols)`
- if data_format='channels_last'
5D tensor with shape:
`(samples,time, rows, cols, channels)`

__Output shape__

- if `return_sequences`
- if data_format='channels_first'
5D tensor with shape:
`(samples, time, filters, output_row, output_col)`
- if data_format='channels_last'
5D tensor with shape:
`(samples, time, output_row, output_col, filters)`
- else
- if data_format ='channels_first'
4D tensor with shape:
`(samples, filters, output_row, output_col)`
- if data_format='channels_last'
4D tensor with shape:
`(samples, output_row, output_col, filters)`
where o_row and o_col depend on the shape of the filter and
the padding

__Raises__

- __ValueError__: in case of invalid constructor arguments.

__References__

- [Convolutional LSTM Network: A Machine Learning Approach for
Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
The current implementation does not include the feedback loop on the
cells output

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L761)</span>
### SimpleRNNCell

```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

Cell class for SimpleRNN.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1132)</span>
### GRUCell

```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

Cell class for the GRU layer.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __recurrent_activation__: Activation function to use
for the recurrent step
(see [activations](../activations.md)).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.
- __implementation__: Implementation mode, either 1 or 2.
Mode 1 will structure its operations as a larger number of
smaller dot products and additions, whereas mode 2 will
batch them into fewer, larger operations. These modes will
have different performance profiles on different hardware and
for different applications.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1610)</span>
### LSTMCell

```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

Cell class for the LSTM layer.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __activation__: Activation function to use
(see [activations](../activations.md)).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
- __recurrent_activation__: Activation function to use
for the recurrent step
(see [activations](../activations.md)).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __unit_forget_bias__: Boolean.
If True, add 1 to the bias of the forget gate at initialization.
Setting it to true will also force `bias_initializer="zeros"`.
This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the inputs.
- __recurrent_dropout__: Float between 0 and 1.
Fraction of the units to drop for
the linear transformation of the recurrent state.
- __implementation__: Implementation mode, either 1 or 2.
Mode 1 will structure its operations as a larger number of
smaller dot products and additions, whereas mode 2 will
batch them into fewer, larger operations. These modes will
have different performance profiles on different hardware and
for different applications.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L20)</span>
### StackedRNNCells

```python
keras.layers.StackedRNNCells(cells)
```

Wrapper allowing a stack of RNN cells to behave as a single cell.

Used to implement efficient stacked RNNs.

__Arguments__

- __cells__: List of RNN cell instances.

__Examples__


```python
cells = [
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
]

inputs = keras.Input((timesteps, input_dim))
x = keras.layers.RNN(cells)(inputs)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L129)</span>
### CuDNNGRU

```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

Fast GRU implementation backed by [CuDNN](https://developer.nvidia.com/cudnn).

Can only be run on GPU, with the TensorFlow backend.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __return_sequences__: Boolean. Whether to return the last output.
in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
in addition to the output.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L318)</span>
### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

Fast LSTM implementation backed by [CuDNN](https://developer.nvidia.com/cudnn).

Can only be run on GPU, with the TensorFlow backend.

__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs.
(see [initializers](../initializers.md)).
- __unit_forget_bias__: Boolean.
If True, add 1 to the bias of the forget gate at initialization.
Setting it to true will also force `bias_initializer="zeros"`.
This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __recurrent_initializer__: Initializer for the `recurrent_kernel`
weights matrix,
used for the linear transformation of the recurrent state.
(see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
(see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
the `kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __recurrent_regularizer__: Regularizer function applied to
the `recurrent_kernel` weights matrix
(see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
(see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
the `kernel` weights matrix
(see [constraints](../constraints.md)).
- __recurrent_constraint__: Constraint function applied to
the `recurrent_kernel` weights matrix
(see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
(see [constraints](../constraints.md)).
- __return_sequences__: Boolean. Whether to return the last output.
in the output sequence, or the full sequence.
- __return_state__: Boolean. Whether to return the last state
in addition to the output.
- __stateful__: Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
