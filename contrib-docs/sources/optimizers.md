
## Available loss functions

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/optimizers/ftml.py#L7)</span>
### FTML

```python
keras_contrib.optimizers.ftml.FTML(lr=0.0025, beta_1=0.6, beta_2=0.999, epsilon=1e-08, decay=0.0)
```

FTML optimizer.

__Arguments__

- __lr__: float >= 0. Learning rate.
- __beta_1__: float, 0 < beta < 1. Generally close to 0.5.
- __beta_2__: float, 0 < beta < 1. Generally close to 1.
- __epsilon__: float >= 0. Fuzz factor.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [FTML - Follow the Moving Leader in Deep Learning](http://www.cse.ust.hk/~szhengac/papers/icml17.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/optimizers/padam.py#L6)</span>
### Padam

```python
keras_contrib.optimizers.padam.Padam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False, partial=0.125)
```

