
## Usage of optimizers

An optimizer is one of the two arguments required for compiling a Keras model:

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

You can either instantiate an optimizer before passing it to `model.compile()` , as in the above example, or you can call it by its name. In the latter case, the default parameters for the optimizer will be used.

```python
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## Parameters common to all Keras optimizers

The parameters `clipnorm` and `clipvalue` can be used with all optimizers to control gradient clipping:

```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L130)</span>
### SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

Stochastic gradient descent optimizer.

Includes support for momentum,
learning rate decay, and Nesterov momentum.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __momentum__: float >= 0. Parameter updates momentum.
- __decay__: float >= 0. Learning rate decay over each update.
- __nesterov__: boolean. Whether to apply Nesterov momentum.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L192)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
```

RMSProp optimizer.

It is recommended to leave the parameters of this optimizer
at their default values
(except the learning rate, which can be freely tuned).

This optimizer is usually a good choice for recurrent
neural networks.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __rho__: float >= 0.
- __epsilon__: float >= 0. Fuzz factor.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L257)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
```

Adagrad optimizer.

It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __epsilon__: float >= 0.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L314)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
```

Adadelta optimizer.

It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __lr__: float >= 0. Learning rate.
It is recommended to leave it at the default value.
- __rho__: float >= 0.
- __epsilon__: float >= 0. Fuzz factor.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L385)</span>
### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
```

Adam optimizer.

Default parameters follow those provided in the original paper.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __beta_1__: float, 0 < beta < 1. Generally close to 1.
- __beta_2__: float, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L457)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
```

Adamax optimizer from Adam paper's Section 7.

It is a variant of Adam based on the infinity norm.
Default parameters follow those provided in the paper.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __beta_1/beta_2__: floats, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L532)</span>
### Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
```

Nesterov Adam optimizer.

Much like Adam is essentially RMSprop with momentum,
Nadam is Adam RMSprop with Nesterov momentum.

Default parameters follow those provided in the paper.
It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __beta_1/beta_2__: floats, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.

__References__

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L616)</span>
### TFOptimizer

```python
keras.optimizers.TFOptimizer(optimizer)
```

Wrapper class for native TensorFlow optimizers.

