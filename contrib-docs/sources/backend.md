# Keras-contrib backends

----

## Backend functions


### conv2d


```python
keras_contrib.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format='channels_first', image_shape=None, filter_shape=None)
```


2D convolution.
__Arguments__

kernel: kernel tensor.
strides: strides tuple.
padding: string, "same" or "valid".
data_format: "tf" or "th". Whether to use Theano or TensorFlow dimension ordering
in inputs/kernels/ouputs.

----

### extract_image_patches


```python
keras_contrib.backend.extract_image_patches(x, ksizes, ssizes, padding='same', data_format='channels_last')
```



Extract the patches from an image
__Parameters__


x : The input image
ksizes : 2-d tuple with the kernel size
ssizes : 2-d tuple with the strides size
padding : 'same' or 'valid'
data_format : 'channels_last' or 'channels_first'

__Returns__

The (k_w,k_h) patches extracted
TF ==> (batch_size,w,h,k_w,k_h,c)
TH ==> (batch_size,w,h,c,k_w,k_h)

----

### depth_to_space


```python
keras_contrib.backend.depth_to_space(input, scale, data_format=None)
```


Uses phase shift algorithm to convert channels/depth for spatial resolution 
----

### moments


```python
keras_contrib.backend.moments(x, axes, shift=None, keep_dims=False)
```


Wrapper over tensorflow backend call 
----

### clip


```python
keras_contrib.backend.clip(x, min_value, max_value)
```


Element-wise value clipping.

If min_value > max_value, clipping range is [min_value,min_value].

__Arguments__

- __x__: Tensor or variable.
- __min_value__: Tensor, float, int, or None.
    If min_value is None, defaults to -infinity.
- __max_value__: Tensor, float, int, or None.
    If max_value is None, defaults to infinity.

__Returns__

A tensor.






