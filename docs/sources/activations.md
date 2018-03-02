
## Usage of activations

Activations can either be used through an `Activation` layer, or through the `activation` argument supported by all forward layers:

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

This is equivalent to:

```python
model.add(Dense(64, activation='tanh'))
```

You can also pass an element-wise TensorFlow/Theano/CNTK function as an activation:

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
model.add(Activation(K.tanh))
```

## Available activations

### softmax


```python
softmax(x, axis=-1)
```


Softmax activation function.

__Arguments__

x : Tensor.
- __axis__: Integer, axis along which the softmax normalization is applied.

__Returns__

Tensor, output of softmax transformation.

__Raises__

- __ValueError__: In case `dim(x) == 1`.

----

### elu


```python
elu(x, alpha=1.0)
```

----

### selu


```python
selu(x)
```


Scaled Exponential Linear Unit. (Klambauer et al., 2017)

__Arguments__

- __x__: A tensor or variable to compute the activation function for.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

### softplus


```python
softplus(x)
```

----

### softsign


```python
softsign(x)
```

----

### relu


```python
relu(x, alpha=0.0, max_value=None)
```

----

### tanh


```python
tanh(x)
```

----

### sigmoid


```python
sigmoid(x)
```

----

### hard_sigmoid


```python
hard_sigmoid(x)
```

----

### linear


```python
linear(x)
```


## On "Advanced Activations"

Activations that are more complex than a simple TensorFlow/Theano/CNTK function (eg. learnable activations, which maintain a state) are available as [Advanced Activation layers](layers/advanced-activations.md), and can be found in the module `keras.layers.advanced_activations`. These include `PReLU` and `LeakyReLU`.
