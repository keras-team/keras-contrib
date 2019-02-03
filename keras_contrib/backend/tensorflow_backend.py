import tensorflow as tf

try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc
import keras.backend as K

py_all = all


def _preprocess_conv2d_input(x, data_format):
    """Transpose and cast the input before the conv2d.

    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """
    if K.dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if data_format == 'channels_first':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = tf.transpose(x, (0, 2, 3, 1))
    return x


def _postprocess_conv2d_output(x, data_format):
    """Transpose and cast the output from conv2d if needed.

    # Arguments
        x: A tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.
    """

    if data_format == 'channels_first':
        x = tf.transpose(x, (0, 3, 1, 2))

    if K.floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.

    # Arguments
        padding: string, `"same"` or `"valid"`.

    # Returns
        a string, `"SAME"` or `"VALID"`.

    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding:', padding)
    return padding


def conv2d(x, kernel, strides=(1, 1), padding='valid', data_format='channels_first',
           image_shape=None, filter_shape=None):
    """2D convolution.

    # Arguments
        x: Input tensor
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: 'channels_first' or 'channels_last'.
            Whether to use Theano or TensorFlow dimension
            ordering in inputs/kernels/ouputs.
        image_shape: Optional, the input tensor shape
        filter_shape: Optional, the kernel shape.

    # Returns
        x convolved with the kernel.

    # Raises
        Exception: In case of invalid border mode or data format.
    """
    return K.conv2d(x, kernel, strides, padding, data_format)


def extract_image_patches(x, ksizes, ssizes, padding='same',
                          data_format='channels_last'):
    """Extract the patches from an image.

    # Arguments
        x: The input image
        ksizes: 2-d tuple with the kernel size
        ssizes: 2-d tuple with the strides size
        padding: 'same' or 'valid'
        data_format: 'channels_last' or 'channels_first'

    # Returns
        The (k_w,k_h) patches extracted
        TF ==> (batch_size,w,h,k_w,k_h,c)
        TH ==> (batch_size,w,h,c,k_w,k_h)
    """
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_padding(padding)
    if data_format == 'channels_first':
        x = K.permute_dimensions(x, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = K.int_shape(x)
    patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                       padding)
    # Reshaping to fit Theano
    bs, w, h, ch = K.int_shape(patches)
    reshaped = tf.reshape(patches, [-1, w, h, tf.floordiv(ch, ch_i), ch_i])
    final_shape = [-1, w, h, ch_i, ksizes[0], ksizes[1]]
    patches = tf.reshape(tf.transpose(reshaped, [0, 1, 2, 4, 3]), final_shape)
    if data_format == 'channels_last':
        patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def depth_to_space(input, scale, data_format=None):
    """ Uses phase shift algorithm to convert channels/depth for spatial resolution.

    # Arguments
        input: Input tensor
        scale: n `int` that is `>= 2`. The size of the spatial block.
        data_format: 'channels_first' or 'channels_last'.
            Whether to use Theano or TensorFlow dimension
            ordering in inputs/kernels/ouputs.

    # Returns
        TODO (PR welcome): Filling this section.
    """
    if data_format is None:
        data_format = K.image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)
    out = tf.depth_to_space(input, scale)
    out = _postprocess_conv2d_output(out, data_format)
    return out


def moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''

    return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)
