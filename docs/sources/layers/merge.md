<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L184)</span>
### Add

```python
keras.layers.Add()
```

Layer that adds a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

__Examples__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L214)</span>
### Subtract

```python
keras.layers.Subtract()
```

Layer that subtracts two inputs.

It takes as input a list of tensors of size 2,
both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
also of the same shape.

__Examples__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L248)</span>
### Multiply

```python
keras.layers.Multiply()
```

Layer that multiplies (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L263)</span>
### Average

```python
keras.layers.Average()
```

Layer that averages a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L278)</span>
### Maximum

```python
keras.layers.Maximum()
```

Layer that computes the maximum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L308)</span>
### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

Layer that concatenates a list of inputs.

It takes as input a list of tensors,
all of the same shape except for the concatenation axis,
and returns a single tensor, the concatenation of all inputs.

__Arguments__

- __axis__: Axis along which to concatenate.
- __**kwargs__: standard layer keyword arguments.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L398)</span>
### Dot

```python
keras.layers.Dot(axes, normalize=False)
```

Layer that computes a dot product between samples in two tensors.

E.g. if applied to two tensors `a` and `b` of shape `(batch_size, n)`,
the output will be a tensor of shape `(batch_size, 1)`
where each entry `i` will be the dot product between
`a[i]` and `b[i]`.

__Arguments__

- __axes__: Integer or tuple of integers,
axis or axes along which to take the dot product.
- __normalize__: Whether to L2-normalize samples along the
dot product axis before taking the dot product.
If set to True, then the output of the dot product
is the cosine proximity between the two samples.
- __**kwargs__: Standard layer keyword arguments.

----

### add


```python
keras.layers.add(inputs)
```


Functional interface to the `Add` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the sum of the inputs.

__Examples__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### subtract


```python
keras.layers.subtract(inputs)
```


Functional interface to the `Subtract` layer.

__Arguments__

- __inputs__: A list of input tensors (exactly 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the difference of the inputs.

__Examples__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### multiply


```python
keras.layers.multiply(inputs)
```


Functional interface to the `Multiply` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise product of the inputs.

----

### average


```python
keras.layers.average(inputs)
```


Functional interface to the `Average` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the average of the inputs.

----

### maximum


```python
keras.layers.maximum(inputs)
```


Functional interface to the `Maximum` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise maximum of the inputs.

----

### concatenate


```python
keras.layers.concatenate(inputs, axis=-1)
```


Functional interface to the `Concatenate` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __axis__: Concatenation axis.
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the concatenation of the inputs alongside axis `axis`.

----

### dot


```python
keras.layers.dot(inputs, axes, normalize=False)
```


Functional interface to the `Dot` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __axes__: Integer or tuple of integers,
axis or axes along which to take the dot product.
- __normalize__: Whether to L2-normalize samples along the
dot product axis before taking the dot product.
If set to True, then the output of the dot product
is the cosine proximity between the two samples.
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the dot product of the samples from the inputs.
