<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization.py#L9)</span>
### InstanceNormalization

```python
keras_contrib.layers.InstanceNormalization(axis=None, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
Normalize the activations of the previous layer at each step,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.
__Arguments__

axis: Integer, the axis that should be normalized
(typically the features axis).
For instance, after a `Conv2D` layer with
`data_format="channels_first"`,
set `axis=1` in `InstanceNormalization`.
Setting `axis=None` will normalize all values in each instance of the batch.
Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
epsilon: Small float added to variance to avoid dividing by zero.
center: If True, add offset of `beta` to normalized tensor.
If False, `beta` is ignored.
scale: If True, multiply by `gamma`.
If False, `gamma` is not used.
When the next layer is linear (also e.g. `nn.relu`),
this can be disabled since the scaling
will be done by the next layer.
beta_initializer: Initializer for the beta weight.
gamma_initializer: Initializer for the gamma weight.
beta_regularizer: Optional regularizer for the beta weight.
gamma_regularizer: Optional regularizer for the gamma weight.
beta_constraint: Optional constraint for the beta weight.
gamma_constraint: Optional constraint for the gamma weight.
__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.
__Output shape__

Same shape as input.
__References__

- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization.py#L149)</span>
### BatchRenormalization

```python
keras_contrib.layers.BatchRenormalization(axis=-1, momentum=0.99, center=True, scale=True, epsilon=0.001, r_max_value=3.0, d_max_value=5.0, t_delta=0.001, weights=None, beta_initializer='zero', gamma_initializer='one', moving_mean_initializer='zeros', moving_variance_initializer='ones', gamma_regularizer=None, beta_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

Batch renormalization layer (Sergey Ioffe, 2017).

Normalize the activations of the previous layer at each batch,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.

__Arguments__

- __axis__: Integer, the axis that should be normalized
    (typically the features axis).
    For instance, after a `Conv2D` layer with
    `data_format="channels_first"`,
    set `axis=1` in `BatchRenormalization`.
- __momentum__: momentum in the computation of the
    exponential average of the mean and standard deviation
    of the data, for feature-wise normalization.
- __center__: If True, add offset of `beta` to normalized tensor.
    If False, `beta` is ignored.
- __scale__: If True, multiply by `gamma`.
    If False, `gamma` is not used.
- __epsilon__: small float > 0. Fuzz parameter.
    Theano expects epsilon >= 1e-5.
- __r_max_value__: Upper limit of the value of r_max.
- __d_max_value__: Upper limit of the value of d_max.
- __t_delta__: At each iteration, increment the value of t by t_delta.
- __weights__: Initialization weights.
    List of 2 Numpy arrays, with shapes:
    `[(input_shape,), (input_shape,)]`
    Note that the order of this list is [gamma, beta, mean, std]
- __beta_initializer__: name of initialization function for shift parameter
    (see [initializers](../initializers.md)), or alternatively,
    Theano/TensorFlow function to use for weights initialization.
    This parameter is only relevant if you don't pass a `weights` argument.
- __gamma_initializer__: name of initialization function for scale parameter (see
    [initializers](../initializers.md)), or alternatively,
    Theano/TensorFlow function to use for weights initialization.
    This parameter is only relevant if you don't pass a `weights` argument.
- __moving_mean_initializer__: Initializer for the moving mean.
- __moving_variance_initializer__: Initializer for the moving variance.
- __gamma_regularizer__: instance of [WeightRegularizer](../regularizers.md)
    (eg. L1 or L2 regularization), applied to the gamma vector.
- __beta_regularizer__: instance of [WeightRegularizer](../regularizers.md),
    applied to the beta vector.
- __beta_constraint__: Optional constraint for the beta weight.
- __gamma_constraint__: Optional constraint for the gamma weight.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization.py#L381)</span>
### GroupNormalization

```python
keras_contrib.layers.GroupNormalization(groups=32, axis=-1, epsilon=1e-05, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

Group normalization layer

Group Normalization divides the channels into groups and computes within each group
the mean and variance for normalization. Group Normalization's computation is independent
of batch sizes, and its accuracy is stable in a wide range of batch sizes.

Relation to Layer Normalization:
If the number of groups is set to 1, then this operation becomes identical to
Layer Normalization.

Relation to Instance Normalization:
If the number of groups is set to the input dimension (number of groups is equal
to number of channels), then this operation becomes identical to Instance Normalization.

__Arguments__

- __groups__: Integer, the number of groups for Group Normalization.
    Can be in the range [1, N] where N is the input dimension.
    The input dimension must be divisible by the number of groups.
- __axis__: Integer, the axis that should be normalized
    (typically the features axis).
    For instance, after a `Conv2D` layer with
    `data_format="channels_first"`,
    set `axis=1` in `BatchNormalization`.
- __epsilon__: Small float added to variance to avoid dividing by zero.
- __center__: If True, add offset of `beta` to normalized tensor.
    If False, `beta` is ignored.
- __scale__: If True, multiply by `gamma`.
    If False, `gamma` is not used.
    When the next layer is linear (also e.g. `nn.relu`),
    this can be disabled since the scaling
    will be done by the next layer.
- __beta_initializer__: Initializer for the beta weight.
- __gamma_initializer__: Initializer for the gamma weight.
- __beta_regularizer__: Optional regularizer for the beta weight.
- __gamma_regularizer__: Optional regularizer for the gamma weight.
- __beta_constraint__: Optional constraint for the beta weight.
- __gamma_constraint__: Optional constraint for the gamma weight.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

__References__

- [Group Normalization](https://arxiv.org/abs/1803.08494)
