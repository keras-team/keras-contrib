# The Sequential model API

To get started, read [this guide to the Keras Sequential model](/getting-started/sequential-model-guide).

## Useful attributes of Model

- `model.layers` is a list of the layers added to the model.


----

## Sequential model methods

### compile


```python
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```


Configures the model for training.

__Arguments__

- __optimizer__: String (name of optimizer) or optimizer object.
See [optimizers](/optimizers).
- __loss__: String (name of objective function) or objective function.
See [losses](/losses).
If the model has multiple outputs, you can use a different loss
on each output by passing a dictionary or a list of losses.
The loss value that will be minimized by the model
will then be the sum of all individual losses.
- __metrics__: List of metrics to be evaluated by the model
during training and testing.
Typically you will use `metrics=['accuracy']`.
To specify different metrics for different outputs of a
multi-output model, you could also pass a dictionary,
such as `metrics={'output_a': 'accuracy'}`.
- __sample_weight_mode__: If you need to do timestep-wise
sample weighting (2D weights), set this to `"temporal"`.
`None` defaults to sample-wise weights (1D).
If the model has multiple outputs, you can use a different
`sample_weight_mode` on each output by passing a
dictionary or a list of modes.
- __weighted_metrics__: List of metrics to be evaluated and weighted
by sample_weight or class_weight during training and testing.
- __target_tensors__: By default, Keras will create a placeholder for the
model's target, which will be fed with the target data during
training. If instead you would like to use your own
target tensor (in turn, Keras will not expect external
Numpy data for these targets at training time), you
can specify them via the `target_tensors` argument.
It should be a single tensor
(for a single-output `Sequential` model).
- __**kwargs__: When using the Theano/CNTK backends, these arguments
are passed into `K.function`.
When using the TensorFlow backend,
these arguments are passed into `tf.Session.run`.

__Raises__

- __ValueError__: In case of invalid arguments for

__Example__

```python
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

----

### fit


```python
fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```


Trains the model for a fixed number of epochs (iterations on a dataset).

__Arguments__

- __x__: Numpy array of training data.
If the input layer in the model is named, you can also pass a
dictionary mapping the input name to a Numpy array.
`x` can be `None` (default) if feeding from
framework-native tensors (e.g. TensorFlow data tensors).
- __y__: Numpy array of target (label) data.
If the output layer in the model is named, you can also pass a
dictionary mapping the output name to a Numpy array.
`y` can be `None` (default) if feeding from
framework-native tensors (e.g. TensorFlow data tensors).
- __batch_size__: Integer or `None`.
Number of samples per gradient update.
If unspecified, it will default to 32.
- __epochs__: Integer. Number of epochs to train the model.
An epoch is an iteration over the entire `x` and `y`
data provided.
Note that in conjunction with `initial_epoch`,
`epochs` is to be understood as "final epoch".
The model is not trained for a number of iterations
given by `epochs`, but merely until the epoch
of index `epochs` is reached.
- __verbose__: 0, 1, or 2. Verbosity mode.
0 = silent, 1 = progress bar, 2 = one line per epoch.
- __callbacks__: List of `keras.callbacks.Callback` instances.
List of callbacks to apply during training.
See [callbacks](/callbacks).
- __validation_split__: Float between 0 and 1.
Fraction of the training data to be used as validation data.
The model will set apart this fraction of the training data,
will not train on it, and will evaluate
the loss and any model metrics
on this data at the end of each epoch.
The validation data is selected from the last samples
in the `x` and `y` data provided, before shuffling.
- __validation_data__: tuple `(x_val, y_val)` or tuple
`(x_val, y_val, val_sample_weights)` on which to evaluate
the loss and any model metrics at the end of each epoch.
The model will not be trained on this data.
This will override `validation_split`.
- __shuffle__: Boolean (whether to shuffle the training data
before each epoch) or str (for 'batch').
'batch' is a special option for dealing with the
limitations of HDF5 data; it shuffles in batch-sized chunks.
Has no effect when `steps_per_epoch` is not `None`.
- __class_weight__: Optional dictionary mapping class indices (integers)
to a weight (float) value, used for weighting the loss function
(during training only).
This can be useful to tell the model to
"pay more attention" to samples from
an under-represented class.
- __sample_weight__: Optional Numpy array of weights for
the training samples, used for weighting the loss function
(during training only). You can either pass a flat (1D)
Numpy array with the same length as the input samples
(1:1 mapping between weights and samples),
or in the case of temporal data,
you can pass a 2D array with shape
`(samples, sequence_length)`,
to apply a different weight to every timestep of every sample.
In this case you should make sure to specify
`sample_weight_mode="temporal"` in `compile()`.
- __initial_epoch__: Epoch at which to start training
(useful for resuming a previous training run).
- __steps_per_epoch__: Total number of steps (batches of samples)
before declaring one epoch finished and starting the
next epoch. When training with input tensors such as
TensorFlow data tensors, the default `None` is equal to
the number of unique samples in your dataset divided by
the batch size, or 1 if that cannot be determined.
- __validation_steps__: Only relevant if `steps_per_epoch`
is specified. Total number of steps (batches of samples)
to validate before stopping.

__Returns__

A `History` object. Its `History.history` attribute is
a record of training loss values and metrics values
at successive epochs, as well as validation loss values
and validation metrics values (if applicable).

__Raises__

- __RuntimeError__: If the model was never compiled.
- __ValueError__: In case of mismatch between the provided input data
and what the model expects.

----

### evaluate


```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```


Computes the loss on some input data, batch by batch.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __batch_size__: integer. Number of samples per gradient update.
- __verbose__: verbosity mode, 0 or 1.
- __sample_weight__: sample weights, as a Numpy array.

__Returns__

Scalar test loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### predict


```python
predict(self, x, batch_size=32, verbose=0)
```


Generates output predictions for the input samples.

The input samples are processed batch by batch.

__Arguments__

- __x__: the input data, as a Numpy array.
- __batch_size__: integer.
- __verbose__: verbosity mode, 0 or 1.

__Returns__

A Numpy array of predictions.

----

### train_on_batch


```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```


Single gradient update over one batch of samples.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __class_weight__: dictionary mapping classes to a weight value,
used for scaling the loss function (during training only).
- __sample_weight__: sample weights, as a Numpy array.

__Returns__

Scalar training loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### test_on_batch


```python
test_on_batch(self, x, y, sample_weight=None)
```


Evaluates the model over a single batch of samples.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
(if the model has multiple inputs).
- __y__: labels, as a Numpy array.
- __sample_weight__: sample weights, as a Numpy array.

__Returns__

Scalar test loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### predict_on_batch


```python
predict_on_batch(self, x)
```


Returns predictions for a single batch of samples.

__Arguments__

- __x__: input data, as a Numpy array or list of Numpy arrays
(if the model has multiple inputs).

__Returns__

A Numpy array of predictions.

----

### fit_generator


```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```


Fits the model on data generated batch-by-batch by a Python generator.

The generator is run in parallel to the model, for efficiency.
For instance, this allows you to do real-time data augmentation
on images on CPU in parallel to training your model on GPU.

__Arguments__

- __generator__: A generator.
The output of the generator must be either
- a tuple (inputs, targets)
- a tuple (inputs, targets, sample_weights).
All arrays should contain the same number of samples.
The generator is expected to loop over its data
indefinitely. An epoch finishes when `steps_per_epoch`
batches have been seen by the model.
- __steps_per_epoch__: Total number of steps (batches of samples)
to yield from `generator` before declaring one epoch
finished and starting the next epoch. It should typically
be equal to the number of unique samples of your dataset
divided by the batch size.
- __epochs__: Integer, total number of iterations on the data.
Note that in conjunction with initial_epoch, the parameter
epochs is to be understood as "final epoch". The model is
not trained for n steps given by epochs, but until the
epoch epochs is reached.
- __verbose__: Verbosity mode, 0, 1, or 2.
- __callbacks__: List of callbacks to be called during training.
- __validation_data__: This can be either
- A generator for the validation data
- A tuple (inputs, targets)
- A tuple (inputs, targets, sample_weights).
- __validation_steps__: Only relevant if `validation_data`
is a generator.
Number of steps to yield from validation generator
at the end of every epoch. It should typically
be equal to the number of unique samples of your
validation dataset divided by the batch size.
- __class_weight__: Dictionary mapping class indices to a weight
for the class.
- __max_queue_size__: Maximum size for the generator queue
- __workers__: Maximum number of processes to spin up
- __use_multiprocessing__: if True, use process based threading.
Note that because
this implementation relies on multiprocessing,
you should not pass
non picklable arguments to the generator
as they can't be passed
easily to children processes.
- __shuffle__: Whether to shuffle the order of the batches at
the beginning of each epoch. Only used with instances
of `Sequence` (keras.utils.Sequence).
- __initial_epoch__: Epoch at which to start training
(useful for resuming a previous training run).

__Returns__

A `History` object.

__Raises__

- __RuntimeError__: if the model was never compiled.

__Example__


```python
def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            yield (x, y)
        f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=1000, epochs=10)
```

----

### evaluate_generator


```python
evaluate_generator(self, generator, steps, max_queue_size=10, workers=1, use_multiprocessing=False)
```


Evaluates the model on a data generator.

The generator should return the same kind of data
as accepted by `test_on_batch`.

__Arguments__

- __generator__: Generator yielding tuples (inputs, targets)
or (inputs, targets, sample_weights)
- __steps__: Total number of steps (batches of samples)
to yield from `generator` before stopping.
- __max_queue_size__: maximum size for the generator queue
- __workers__: maximum number of processes to spin up
- __use_multiprocessing__: if True, use process based threading.
Note that because this implementation
relies on multiprocessing, you should not pass
non picklable arguments to the generator
as they can't be passed easily to children processes.

__Returns__

Scalar test loss (if the model has no metrics)
or list of scalars (if the model computes other metrics).
The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

__Raises__

- __RuntimeError__: if the model was never compiled.

----

### predict_generator


```python
predict_generator(self, generator, steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```


Generates predictions for the input samples from a data generator.

The generator should return the same kind of data as accepted by
`predict_on_batch`.

__Arguments__

- __generator__: generator yielding batches of input samples.
- __steps__: Total number of steps (batches of samples)
to yield from `generator` before stopping.
- __max_queue_size__: maximum size for the generator queue
- __workers__: maximum number of processes to spin up
- __use_multiprocessing__: if True, use process based threading.
Note that because this implementation
relies on multiprocessing, you should not pass
non picklable arguments to the generator
as they can't be passed easily to children processes.
- __verbose__: verbosity mode, 0 or 1.

__Returns__

A Numpy array of predictions.

----

### get_layer


```python
get_layer(self, name=None, index=None)
```


Retrieve a layer that is part of the model.

Returns a layer based on either its name (unique)
or its index in the graph. Indices are based on
order of horizontal graph traversal (bottom-up).

__Arguments__

- __name__: string, name of layer.
- __index__: integer, index of layer.

__Returns__

A layer instance.
