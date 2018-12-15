<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/crf.py#L15)</span>
### CRF

```python
keras_contrib.layers.CRF(units, learn_mode='join', test_mode=None, sparse_target=False, use_boundary=True, use_bias=True, activation='linear', kernel_initializer='glorot_uniform', chain_initializer='orthogonal', bias_initializer='zeros', boundary_initializer='zeros', kernel_regularizer=None, chain_regularizer=None, boundary_regularizer=None, bias_regularizer=None, kernel_constraint=None, chain_constraint=None, boundary_constraint=None, bias_constraint=None, input_dim=None, unroll=False)
```

An implementation of linear chain conditional random field (CRF).

An linear chain CRF is defined to maximize the following likelihood function:

$$ L(W, U, b; y_1, ..., y_n) := rac{1}{Z} \sum_{y_1, ..., y_n} \exp(-a_1' y_1 - a_n' y_n
- \sum_{k=1^n}((f(x_k' W + b) y_k) + y_1' U y_2)), $$

where:
$Z$: normalization constant
$x_k, y_k$:  inputs and outputs

This implementation has two modes for optimization:
1. (`join mode`) optimized by maximizing join likelihood, which is optimal in theory of statistics.
Note that in this case, CRF must be the output/last layer.
2. (`marginal mode`) return marginal probabilities on each time step and optimized via composition
likelihood (product of marginal likelihood), i.e., using `categorical_crossentropy` loss.
Note that in this case, CRF can be either the last layer or an intermediate layer (though not explored).

For prediction (test phrase), one can choose either Viterbi best path (class indices) or marginal
probabilities if probabilities are needed. However, if one chooses *join mode* for training,
Viterbi output is typically better than marginal output, but the marginal output will still perform
reasonably close, while if *marginal mode* is used for training, marginal output usually performs
much better. The default behavior is set according to this observation.

In addition, this implementation supports masking and accepts either onehot or sparse target.


__Examples__


```python
model = Sequential()
model.add(Embedding(3001, 300, mask_zero=True)(X)

# use learn_mode = 'join', test_mode = 'viterbi', sparse_target = True (label indice output)
crf = CRF(10, sparse_target=True)
model.add(crf)

# crf.accuracy is default to Viterbi acc if using join-mode (default).
# One can add crf.marginal_acc if interested, but may slow down learning
model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

# y must be label indices (with shape 1 at dim 3) here, since `sparse_target=True`
model.fit(x, y)

# prediction give onehot representation of Viterbi best path
y_hat = model.predict(x_test)
```


__Arguments__

- __units__: Positive integer, dimensionality of the output space.
- __learn_mode__: Either 'join' or 'marginal'.
    The former train the model by maximizing join likelihood while the latter
    maximize the product of marginal likelihood over all time steps.
- __test_mode__: Either 'viterbi' or 'marginal'.
    The former is recommended and as default when `learn_mode = 'join'` and
    gives one-hot representation of the best path at test (prediction) time,
    while the latter is recommended and chosen as default when `learn_mode = 'marginal'`,
    which produces marginal probabilities for each time step.
- __sparse_target__: Boolean (default False) indicating if provided labels are one-hot or
    indices (with shape 1 at dim 3).
- __use_boundary__: Boolean (default True) indicating if trainable start-end chain energies
    should be added to model.
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix,
    used for the linear transformation of the inputs.
    (see [initializers](../initializers.md)).
- __chain_initializer__: Initializer for the `chain_kernel` weights matrix,
    used for the CRF chain energy.
    (see [initializers](../initializers.md)).
- __boundary_initializer__: Initializer for the `left_boundary`, 'right_boundary' weights vectors,
    used for the start/left and end/right boundary energy.
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __activation__: Activation function to use
    (see [activations](../activations.md)).
    If you pass None, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __chain_regularizer__: Regularizer function applied to
    the `chain_kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __boundary_regularizer__: Regularizer function applied to
    the 'left_boundary', 'right_boundary' weight vectors
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to
    the `kernel` weights matrix
    (see [constraints](../constraints.md)).
- __chain_constraint__: Constraint function applied to
    the `chain_kernel` weights matrix
    (see [constraints](../constraints.md)).
- __boundary_constraint__: Constraint function applied to
    the `left_boundary`, `right_boundary` weights vectors
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).
- __input_dim__: dimensionality of the input (integer).
    This argument (or alternatively, the keyword argument `input_shape`)
    is required when using this layer as the first layer in a model.
- __unroll__: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used.
    Unrolling can speed-up a RNN, although it tends to be more memory-intensive.
    Unrolling is only suitable for short sequences.

__Input shape__

3D tensor with shape `(nb_samples, timesteps, input_dim)`.

__Output shape__

3D tensor with shape `(nb_samples, timesteps, units)`.

__Masking__

This layer supports masking for input data with a variable number
of timesteps. To introduce masks to your data,
use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
set to `True`.

