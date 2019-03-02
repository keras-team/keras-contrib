<h1 id="keras_contrib.layers.InstanceNormalization">InstanceNormalization</h1>

```python
InstanceNormalization(self, axis=None, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)
```
Instance normalization layer.

Normalize the activations of the previous layer at each step,
i.e. applies a transformation that maintains the mean activation
close to 0 and the activation standard deviation close to 1.

__Arguments__

- __axis__: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `InstanceNormalization`.
        Setting `axis=None` will normalize all values in each
        instance of the batch.
        Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
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
    when using this layer as the first layer in a Sequential model.

__Output shape__

    Same shape as input.

__References__

    - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    - [Instance Normalization: The Missing Ingredient for Fast Stylization](
    https://arxiv.org/abs/1607.08022)

<h1 id="keras_contrib.layers.GroupNormalization">GroupNormalization</h1>

```python
GroupNormalization(self, groups=32, axis=-1, epsilon=1e-05, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)
```
Group normalization layer.

Group Normalization divides the channels into groups and computes
within each group
the mean and variance for normalization.
Group Normalization's computation is independent
 of batch sizes, and its accuracy is stable in a wide range of batch sizes.

Relation to Layer Normalization:
If the number of groups is set to 1, then this operation becomes identical to
Layer Normalization.

Relation to Instance Normalization:
If the number of groups is set to the
input dimension (number of groups is equal
to number of channels), then this operation becomes
identical to Instance Normalization.

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

