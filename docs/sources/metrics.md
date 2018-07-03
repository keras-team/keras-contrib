
## Usage of metrics

A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the `metrics` parameter when a model is compiled.

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

A metric function is similar to a [loss function](/losses), except that the results from evaluating a metric are not used when training the model.

You can either pass the name of an existing metric, or pass a Theano/TensorFlow symbolic function (see [Custom metrics](#custom-metrics)).

#### Arguments
  - __y_true__: True labels. Theano/TensorFlow tensor.
  - __y_pred__: Predictions. Theano/TensorFlow tensor of the same shape as y_true.

#### Returns
  Single tensor value representing the mean of the output array across all
  datapoints.

----

## Available metrics


### binary_accuracy


```python
binary_accuracy(y_true, y_pred)
```

----

### categorical_accuracy


```python
categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```


----

## Custom metrics

Custom metrics can be passed at the compilation step. The
function would need to take `(y_true, y_pred)` as arguments and return
a single tensor value.

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
