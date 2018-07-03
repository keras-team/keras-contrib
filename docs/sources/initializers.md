## Usage of initializers

Initializations define the way to set the initial random weights of Keras layers.

The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply `kernel_initializer` and `bias_initializer`:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## Available initializers

The following built-in initializers are available as part of the `keras.initializers` module:

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L223)</span>
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

Initializer that generates a random orthogonal matrix.

__Arguments__

- __gain__: Multiplicative factor to apply to the orthogonal matrix.
- __seed__: A Python integer. Used to seed the random generator.

__References__

Saxe et al., http://arxiv.org/abs/1312.6120

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L260)</span>
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

Initializer that generates the identity matrix.

Only use for square 2D matrices.

__Arguments__

- __gain__: Multiplicative factor to apply to the identity matrix.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L9)</span>
### Initializer

```python
keras.initializers.Initializer()
```

Initializer base class: all initializers inherit from this class.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L28)</span>
### Zeros

```python
keras.initializers.Zeros()
```

Initializer that generates tensors initialized to 0.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L36)</span>
### Ones

```python
keras.initializers.Ones()
```

Initializer that generates tensors initialized to 1.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L44)</span>
### Constant

```python
keras.initializers.Constant(value=0)
```

Initializer that generates tensors initialized to a constant value.

__Arguments__

- __value__: float; the value of the generator tensors.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L61)</span>
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

Initializer that generates tensors with a normal distribution.

__Arguments__

- __mean__: a python scalar or a scalar tensor. Mean of the random values
to generate.
- __stddev__: a python scalar or a scalar tensor. Standard deviation of the
random values to generate.
- __seed__: A Python integer. Used to seed the random generator.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L89)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

Initializer that generates tensors with a uniform distribution.

__Arguments__

- __minval__: A python scalar or a scalar tensor. Lower bound of the range
of random values to generate.
- __maxval__: A python scalar or a scalar tensor. Upper bound of the range
of random values to generate.  Defaults to 1 for float types.
- __seed__: A Python integer. Used to seed the random generator.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L117)</span>
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

Initializer that generates a truncated normal distribution.

These values are similar to values from a `RandomNormal`
except that values more than two standard deviations from the mean
are discarded and re-drawn. This is the recommended initializer for
neural network weights and filters.

__Arguments__

- __mean__: a python scalar or a scalar tensor. Mean of the random values
to generate.
- __stddev__: a python scalar or a scalar tensor. Standard deviation of the
random values to generate.
- __seed__: A Python integer. Used to seed the random generator.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L150)</span>
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

Initializer capable of adapting its scale to the shape of weights.

With `distribution="normal"`, samples are drawn from a truncated normal
distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

- number of input units in the weight tensor, if mode = "fan_in"
- number of output units, if mode = "fan_out"
- average of the numbers of input and output units, if mode = "fan_avg"

With `distribution="uniform"`,
samples are drawn from a uniform distribution
within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

__Arguments__

- __scale__: Scaling factor (positive float).
- __mode__: One of "fan_in", "fan_out", "fan_avg".
- __distribution__: Random distribution to use. One of "normal", "uniform".
- __seed__: A Python integer. Used to seed the random generator.

__Raises__

- __ValueError__: In case of an invalid value for the "scale", mode" or
"distribution" arguments.

----

### lecun_uniform


```python
lecun_uniform(seed=None)
```


LeCun uniform initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(3 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

LeCun 98, Efficient Backprop,
- __http__://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

----

### glorot_normal


```python
glorot_normal(seed=None)
```


Glorot normal initializer, also called Xavier normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(2 / (fan_in + fan_out))`
where `fan_in` is the number of input units in the weight tensor
and `fan_out` is the number of output units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

Glorot & Bengio, AISTATS 2010
- __http__://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

----

### glorot_uniform


```python
glorot_uniform(seed=None)
```


Glorot uniform initializer, also called Xavier uniform initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(6 / (fan_in + fan_out))`
where `fan_in` is the number of input units in the weight tensor
and `fan_out` is the number of output units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

Glorot & Bengio, AISTATS 2010
- __http__://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

----

### he_normal


```python
he_normal(seed=None)
```


He normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(2 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

He et al., http://arxiv.org/abs/1502.01852

----

### lecun_normal


```python
lecun_normal(seed=None)
```


LeCun normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(1 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

----

### he_uniform


```python
he_uniform(seed=None)
```


He uniform variance scaling initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(6 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

He et al., http://arxiv.org/abs/1502.01852



An initializer may be passed as a string (must match one of the available initializers above), or as a callable:

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```


## Using custom initializers

If passing a custom callable, then it must take the argument `shape` (shape of the variable to initialize) and `dtype` (dtype of generated values):

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```
