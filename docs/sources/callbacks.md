## Usage of callbacks

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` or `Model` classes. The relevant methods of the callbacks will then be called at each stage of the training. 

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L992)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

Callback for creating simple, custom callbacks on-the-fly.

This callback is constructed with anonymous functions that will be called
at the appropriate time. Note that the callbacks expects positional
arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
`epoch`, `logs`
- `on_batch_begin` and `on_batch_end` expect two positional arguments:
`batch`, `logs`
- `on_train_begin` and `on_train_end` expect one positional argument:
`logs`

__Arguments__

- __on_epoch_begin__: called at the beginning of every epoch.
- __on_epoch_end__: called at the end of every epoch.
- __on_batch_begin__: called at the beginning of every batch.
- __on_batch_end__: called at the end of every batch.
- __on_train_begin__: called at the beginning of model training.
- __on_train_end__: called at the end of model training.

__Example__

```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L142)</span>
### Callback

```python
keras.callbacks.Callback()
```

Abstract base class used to build new callbacks.

__Properties__

- __params__: dict. Training parameters
(eg. verbosity, batch size, number of epochs...).
- __model__: instance of `keras.models.Model`.
Reference of the model being trained.

The `logs` dictionary that callback methods
take as argument will contain keys for quantities relevant to
the current batch or epoch.

Currently, the `.fit()` method of the `Sequential` model class
will include the following quantities in the `logs` that
it passes to its callbacks:

- __on_epoch_end__: logs include `acc` and `loss`, and
optionally include `val_loss`
(if validation is enabled in `fit`), and `val_acc`
(if validation and accuracy monitoring are enabled).
- __on_batch_begin__: logs include `size`,
the number of samples in the current batch.
- __on_batch_end__: logs include `loss`, and optionally `acc`
(if accuracy monitoring is enabled).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L802)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

__Example__

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__Arguments__

- __monitor__: quantity to be monitored.
- __factor__: factor by which the learning rate will
be reduced. new_lr = lr * factor
- __patience__: number of epochs with no improvement
after which learning rate will be reduced.
- __verbose__: int. 0: quiet, 1: update messages.
- __mode__: one of {auto, min, max}. In `min` mode,
lr will be reduced when the quantity
monitored has stopped decreasing; in `max`
mode it will be reduced when the quantity
monitored has stopped increasing; in `auto`
mode, the direction is automatically inferred
from the name of the monitored quantity.
- __epsilon__: threshold for measuring the new optimum,
to only focus on significant changes.
- __cooldown__: number of epochs to wait before resuming
normal operation after lr has been reduced.
- __min_lr__: lower bound on the learning rate.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L917)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

Callback that streams epoch results to a csv file.

Supports all values that can be represented as a string,
including 1D iterables such as np.ndarray.

__Example__

```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__Arguments__

- __filename__: filename of the csv file, e.g. 'run/log.csv'.
- __separator__: string used to separate elements in the csv file.
- __append__: True: append if file exists (useful for continuing
training). False: overwrite existing file,

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L428)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
```

Stop training when a monitored quantity has stopped improving.

__Arguments__

- __monitor__: quantity to be monitored.
- __min_delta__: minimum change in the monitored quantity
to qualify as an improvement, i.e. an absolute
change of less than min_delta, will count as no
improvement.
- __patience__: number of epochs with no improvement
after which training will be stopped.
- __verbose__: verbosity mode.
- __mode__: one of {auto, min, max}. In `min` mode,
training will stop when the quantity
monitored has stopped decreasing; in `max`
mode it will stop when the quantity
monitored has stopped increasing; in `auto`
mode, the direction is automatically inferred
from the name of the monitored quantity.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L226)</span>
### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```

Callback that terminates training when a NaN loss is encountered.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L197)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger()
```

Callback that accumulates epoch averages of metrics.

This callback is automatically applied to every Keras model.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L242)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples')
```

Callback that prints metrics to stdout.

__Arguments__

- __count_mode__: One of "steps" or "samples".
Whether the progress bar should
count samples seen or steps (batches) seen.

__Raises__

- __ValueError__: In case of invalid `count_mode`.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L309)</span>
### History

```python
keras.callbacks.History()
```

Callback that records events into a `History` object.

This callback is automatically applied to
every Keras model. The `History` object
gets returned by the `fit` method of models.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L328)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

Save the model after every epoch.

`filepath` can contain named formatting options,
which will be filled the value of `epoch` and
keys in `logs` (passed in `on_epoch_end`).

For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
then the model checkpoints will be saved with the epoch number and
the validation loss in the filename.

__Arguments__

- __filepath__: string, path to save the model file.
- __monitor__: quantity to monitor.
- __verbose__: verbosity mode, 0 or 1.
- __save_best_only__: if `save_best_only=True`,
the latest best model according to
the quantity monitored will not be overwritten.
- __mode__: one of {auto, min, max}.
If `save_best_only=True`, the decision
to overwrite the current save file is made
based on either the maximization or the
minimization of the monitored quantity. For `val_acc`,
this should be `max`, for `val_loss` this should
be `min`, etc. In `auto` mode, the direction is
automatically inferred from the name of the monitored quantity.
- __save_weights_only__: if True, then only the model's weights will be
saved (`model.save_weights(filepath)`), else the full model
is saved (`model.save(filepath)`).
- __period__: Interval (number of epochs) between checkpoints.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L510)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
```

Callback used to stream events to a server.

Requires the `requests` library.
Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
HTTP POST, with a `data` argument which is a
JSON-encoded dictionary of event data.

__Arguments__

- __root__: String; root url of the target server.
- __path__: String; path relative to `root` to which the events will be sent.
- __field__: String; JSON field under which the data will be stored.
- __headers__: Dictionary; optional custom HTTP headers.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L555)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule)
```

Learning rate scheduler.

__Arguments__

- __schedule__: a function that takes an epoch index as input
(integer, indexed from 0) and returns a new
learning rate as output (float).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L578)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
```

Tensorboard basic visualizations.

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
is a visualization tool provided with TensorFlow.

This callback writes a log for TensorBoard, which allows
you to visualize dynamic graphs of your training and test
metrics, as well as activation histograms for the different
layers in your model.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:
```sh
tensorboard --logdir=/full_path_to_your_logs
```

__Arguments__

- __log_dir__: the path of the directory where to save the log
files to be parsed by TensorBoard.
- __histogram_freq__: frequency (in epochs) at which to compute activation
and weight histograms for the layers of the model. If set to 0,
histograms won't be computed. Validation data (or split) must be
specified for histogram visualizations.
- __write_graph__: whether to visualize the graph in TensorBoard.
The log file can become quite large when
write_graph is set to True.
- __write_grads__: whether to visualize gradient histograms in TensorBoard.
`histogram_freq` must be greater than 0.
- __batch_size__: size of batch of inputs to feed to the network
for histograms computation.
- __write_images__: whether to write model weights to visualize as
image in TensorBoard.
- __embeddings_freq__: frequency (in epochs) at which selected embedding
layers will be saved.
- __embeddings_layer_names__: a list of names of layers to keep eye on. If
None or empty list all the embedding layer will be watched.
- __embeddings_metadata__: a dictionary which maps layer name to a file name
in which metadata for this embedding layer is saved. See the
[details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
about metadata files format. In case if the same metadata file is
used for all embedding layers, string can be passed.


---


# Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`.

Here's a simple example saving a list of losses over each batch during training:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### Example: recording loss history

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### Example: model checkpoints

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```
