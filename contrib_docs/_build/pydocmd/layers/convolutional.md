<h1 id="keras_contrib.layers.CosineConv2D">CosineConvolution2D</h1>

```python
CosineConvolution2D(self, filters, kernel_size, kernel_initializer='glorot_uniform', activation=None, weights=None, padding='valid', strides=(1, 1), data_format=None, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, **kwargs)
```
Cosine Normalized Convolution operator for filtering
windows of two-dimensional inputs.

__Examples__


```python
    # apply a 3x3 convolution with 64 output filters on a 256x256 image:
    model = Sequential()
    model.add(CosineConvolution2D(64, 3, 3,
                            padding='same',
                            input_shape=(3, 256, 256)))
    # now model.output_shape == (None, 64, 256, 256)

    # add a 3x3 convolution on top, with 32 output filters:
    model.add(CosineConvolution2D(32, 3, 3, padding='same'))
    # now model.output_shape == (None, 32, 256, 256)
```

__Arguments__

- __filters__: Number of convolution filters to use.
- __kernel_size__: kernel_size: An integer or tuple/list of
        2 integers, specifying the
        dimensions of the convolution window.
- __init__: name of initialization function for the weights of the layer
- __(see [initializers](https__://keras.io/initializers)), or alternatively,
        Theano function to use for weights initialization.
        This parameter is only relevant if you don't pass
        a `weights` argument.
- __activation__: name of activation function to use
- __(see [activations](https__://keras.io/activations)),
        or alternatively, elementwise Theano function.
        If you don't specify anything, no activation is applied
- __(ie. "linear" activation__: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __padding__: 'valid', 'same' or 'full'
        ('full' requires the Theano backend).
- __strides__: tuple of length 2. Factor by which to strides output.
        Also called strides elsewhere.
- __kernel_regularizer__: instance of [WeightRegularizer](
- __https__://keras.io/regularizers)
        (eg. L1 or L2 regularization), applied to the main weights matrix.
- __bias_regularizer__: instance of [WeightRegularizer](
- __https__://keras.io/regularizers), applied to the use_bias.
- __activity_regularizer__: instance of [ActivityRegularizer](
- __https__://keras.io/regularizers), applied to the network output.
- __kernel_constraint__: instance of the [constraints](
- __https__://keras.io/constraints) module
        (eg. maxnorm, nonneg), applied to the main weights matrix.
- __bias_constraint__: instance of the [constraints](
- __https__://keras.io/constraints) module, applied to the use_bias.
- __data_format__: 'channels_first' or 'channels_last'.
        In 'channels_first' mode, the channels dimension
        (the depth) is at index 1, in 'channels_last' mode is it at index 3.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be `'channels_last'`.
- __use_bias__: whether to include a use_bias
        (i.e. make the layer affine rather than linear).

__Input shape__

    4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

__Output shape__

    4D tensor with shape:
    `(samples, filters, nekernel_rows, nekernel_cols)`
    if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, nekernel_rows, nekernel_cols, filters)`
    if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.


__References__

    - [Cosine Normalization: Using Cosine Similarity Instead
       of Dot Product in Neural Networks](https://arxiv.org/pdf/1702.05870.pdf)

<h1 id="keras_contrib.layers.SubPixelUpscaling">SubPixelUpscaling</h1>

```python
SubPixelUpscaling(self, scale_factor=2, data_format=None, **kwargs)
```
Sub-pixel convolutional upscaling layer.

This layer requires a Convolution2D prior to it,
having output filters computed according to
the formula :

    filters = k * (scale_factor * scale_factor)
    where k = a user defined number of filters (generally larger than 32)
          scale_factor = the upscaling factor (generally 2)

This layer performs the depth to space operation on
the convolution filters, and returns a
tensor with the size as defined below.

__Example :__

```python
    # A standard subpixel upscaling block
    x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
    u = SubPixelUpscaling(scale_factor=2)(x)

    # Optional
    x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
```

In practice, it is useful to have a second convolution layer after the
SubPixelUpscaling layer to speed up the learning process.

However, if you are stacking multiple
SubPixelUpscaling blocks, it may increase
the number of parameters greatly, so the
Convolution layer after SubPixelUpscaling
layer can be removed.

__Arguments__

- __scale_factor__: Upscaling factor.
- __data_format__: Can be None, 'channels_first' or 'channels_last'.

__Input shape__

    4D tensor with shape:
    `(samples, k * (scale_factor * scale_factor) channels, rows, cols)`
    if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, k * (scale_factor * scale_factor) channels)`
    if data_format='channels_last'.

__Output shape__

    4D tensor with shape:
    `(samples, k channels, rows * scale_factor, cols * scale_factor))`
    if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows * scale_factor, cols * scale_factor, k channels)`
    if data_format='channels_last'.

__References__

    - [Real-Time Single Image and Video Super-Resolution Using an
       Efficient Sub-Pixel Convolutional Neural Network](
       https://arxiv.org/abs/1609.05158)

