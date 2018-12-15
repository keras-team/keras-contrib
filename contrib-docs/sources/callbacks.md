## Usage of callbacks

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` or `Model` classes. The relevant methods of the callbacks will then be called at each stage of the training. 

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/snapshot.py#L44)</span>
### SnapshotCallbackBuilder

```python
keras_contrib.callbacks.snapshot.SnapshotCallbackBuilder(nb_epochs, nb_snapshots, init_lr=0.1)
```

Callback builder for snapshot ensemble training of a model.
From the paper "Snapshot Ensembles: Train 1, Get M For Free" (https://openreview.net/pdf?id=BJYwwY9ll)

Creates a list of callbacks, which are provided when training a model
so as to save the model weights at certain epochs, and then sharply
increase the learning rate.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/dead_relu_detector.py#L8)</span>
### DeadReluDetector

```python
keras_contrib.callbacks.dead_relu_detector.DeadReluDetector(x_train, verbose=False)
```

Reports the number of dead ReLUs after each training epoch
ReLU is considered to be dead if it did not fire once for entire training set

__Arguments__

x_train: Training dataset to check whether or not neurons fire
verbose: verbosity mode
True means that even a single dead neuron triggers a warning message
False means that only significant number of dead neurons (10% or more)
triggers a warning message

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/snapshot.py#L16)</span>
### SnapshotModelCheckpoint

```python
keras_contrib.callbacks.snapshot.SnapshotModelCheckpoint(nb_epochs, nb_snapshots, fn_prefix='Model')
```

Callback that saves the snapshot weights of the model.

Saves the model weights on certain epochs (which can be considered the
snapshot of the model at that epoch).

Should be used with the cosine annealing learning rate schedule to save
the weight just before learning rate is sharply increased.

__Arguments:__

nb_epochs: total number of epochs that the model will be trained for.
nb_snapshots: number of times the weights of the model will be saved.
fn_prefix: prefix for the filename of the weights.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_learning_rate.py#L4)</span>
### CyclicLR

```python
keras_contrib.callbacks.cyclical_learning_rate.CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000.0, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle')
```

This callback implements a cyclical learning rate policy (CLR).
The method cycles the learning rate between two boundaries with
some constant frequency.
__Arguments__

- __base_lr__: initial learning rate which is the
    lower boundary in the cycle.
- __max_lr__: upper boundary in the cycle. Functionally,
    it defines the cycle amplitude (max_lr - base_lr).
    The lr at any cycle is the sum of base_lr
    and some scaling of the amplitude; therefore
    max_lr may not actually be reached depending on
    scaling function.
- __step_size__: number of training iterations per
    half cycle. Authors suggest setting step_size
    2-8 x training iterations in epoch.
- __mode__: one of {triangular, triangular2, exp_range}.
    Default 'triangular'.
    Values correspond to policies detailed above.
    If scale_fn is not None, this argument is ignored.
- __gamma__: constant in 'exp_range' scaling function:
    gamma**(cycle iterations)
- __scale_fn__: Custom scaling policy defined by a single
    argument lambda function, where
    0 <= scale_fn(x) <= 1 for all x >= 0.
    mode paramater is ignored
- __scale_mode__: {'cycle', 'iterations'}.
    Defines whether scale_fn is evaluated on
    cycle number or cycle iterations (training
    iterations since start of cycle). Default is 'cycle'.

The amplitude of the cycle can be scaled on a per-iteration or
per-cycle basis.
This class has three built-in policies, as put forth in the paper.
"triangular":
A basic triangular cycle w/ no amplitude scaling.
"triangular2":
A basic triangular cycle that scales initial amplitude by half each cycle.
"exp_range":
A cycle that scales initial amplitude by gamma**(cycle iterations) at each
cycle iteration.
For more detail, please see paper.

__Example for CIFAR-10 w/ batch size 100:__

```python
clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                    step_size=2000., mode='triangular')
model.fit(X_train, Y_train, callbacks=[clr])
```

Class also supports custom scaling functions:
```python
clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                    step_size=2000., scale_fn=clr_fn,
                    scale_mode='cycle')
model.fit(X_train, Y_train, callbacks=[clr])
```

__References__


- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)


---
