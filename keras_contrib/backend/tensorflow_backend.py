import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables

try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc
from keras.backend import tensorflow_backend as KTF
from keras.backend import dtype
from keras.backend.common import floatx
from keras.backend.common import image_data_format
from keras.backend.tensorflow_backend import _to_tensor

py_all = all


def _preprocess_conv2d_input(x, data_format):
    """Transpose and cast the input before the conv2d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    if dtype(x) == 'float64':
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

    if floatx() == 'float64':
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
    '''2D convolution.
    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "tf" or "th". Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    '''
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise Exception('Invalid border mode: ' + str(padding))

    strides = (1,) + strides + (1,)

    if floatx() == 'float64':
        # tf conv2d only supports float32
        x = tf.cast(x, 'float32')
        kernel = tf.cast(kernel, 'float32')

    if data_format == 'channels_first':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = tf.transpose(x, (0, 2, 3, 1))
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
        x = tf.transpose(x, (0, 3, 1, 2))
    elif data_format == 'channels_last':
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
    else:
        raise Exception('Unknown data_format: ' + str(data_format))

    if floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x


def extract_image_patches(x, ksizes, ssizes, padding='same',
                          data_format='channels_last'):
    '''
    Extract the patches from an image
    # Parameters

        x : The input image
        ksizes : 2-d tuple with the kernel size
        ssizes : 2-d tuple with the strides size
        padding : 'same' or 'valid'
        data_format : 'channels_last' or 'channels_first'

    # Returns
        The (k_w,k_h) patches extracted
        TF ==> (batch_size,w,h,k_w,k_h,c)
        TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_padding(padding)
    if data_format == 'channels_first':
        x = KTF.permute_dimensions(x, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = KTF.int_shape(x)
    patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                       padding)
    # Reshaping to fit Theano
    bs, w, h, ch = KTF.int_shape(patches)
    patches = tf.reshape(tf.transpose(tf.reshape(patches, [-1, w, h, tf.floordiv(ch, ch_i), ch_i]), [0, 1, 2, 4, 3]),
                         [-1, w, h, ch_i, ksizes[0], ksizes[1]])
    if data_format == 'channels_last':
        patches = KTF.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def depth_to_space(input, scale, data_format=None):
    ''' Uses phase shift algorithm to convert channels/depth for spatial resolution '''
    if data_format is None:
        data_format = image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)
    out = tf.depth_to_space(input, scale)
    out = _postprocess_conv2d_output(out, data_format)
    return out


def moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''

    return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)


def clip(x, min_value, max_value):
    """Element-wise value clipping.

    If min_value > max_value, clipping range is [min_value,min_value].

    # Arguments
        x: Tensor or variable.
        min_value: Tensor, float, int, or None.
            If min_value is None, defaults to -infinity.
        max_value: Tensor, float, int, or None.
            If max_value is None, defaults to infinity.

    # Returns
        A tensor.
    """
    if max_value is None:
        max_value = np.inf
    if min_value is None:
        min_value = -np.inf
    min_value = _to_tensor(min_value, x.dtype.base_dtype)
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    max_value = tf.maximum(min_value, max_value)
    return tf.clip_by_value(x, min_value, max_value)


def spatial_activation2d(features,
                         activation=tf.nn.softmax,
                         temperature=None,
                         name=None,
                         variables_collections=None,
                         trainable=True,
                         data_format=None):
    """Compute the spatial softmax or another activation of a convolutional feature map.

    This is from the TensorFlow call `spatial_softmax`, which is not in a
    release as of TensorFlow 1.3. Please replace this function with the
    upstream TensorFlow information when it becomes available.

    First computes the softmax over the spatial extent of each channel of a
    convolutional feature map. Then computes the expected 2D position of the
    points of maximal activation for each channel, resulting in a set of
    feature keypoints [x1, y1, ... xN, yN] for all N channels.

    Read more here:
    "Learning visual feature spaces for robotic manipulation with
    deep spatial autoencoders." Finn et. al, http://arxiv.org/abs/1509.06113.

    # Arguments
      features: A `Tensor` of size [batch_size, W, H, num_channels]; the
        convolutional feature map.
      activation: Activation function. The default value is a softmax function.
        Explicitly set it to None to skip it and maintain a linear activation.
      temperature: Softmax temperature (optional). If None, a learnable
        temperature is created.
      name: A name for this operation (optional).
      variables_collections: Collections for the temperature variable.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      data_format: A string. `NHWC` (default) and `NCHW` are supported.
    Returns:
      feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
        the expected 2D locations of each channel's feature keypoint (normalized
        to the range (-1,1)). The inner dimension is arranged as
        [x1, y1, ... xN, yN].
    Raises:
      ValueError: If unexpected data_format specified.
    """
    shape = tf.shape(features)
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_last':
        height, width, num_channels = shape[1], shape[2], shape[3]
    if data_format == 'channels_first':
        num_channels, height, width = shape[1], shape[2], shape[3]
    else:
        raise ValueError('data_format has to be either NCHW or NHWC.')

    with tf.name_scope(name, 'spatial_softmax', [features]) as name:
        # Create tensors for x and y coordinate values, scaled to range [-1, 1].
        pos_x, pos_y = tf.meshgrid(tf.lin_space(-1., 1., num=height),
                                   tf.lin_space(-1., 1., num=width),
                                   indexing='ij')
        pos_x = tf.reshape(pos_x, [height * width])
        pos_y = tf.reshape(pos_y, [height * width])
        if temperature is None:
            temperature_collections = utils.get_variable_collections(
                variables_collections, 'temperature')
            temperature = variables.model_variable(
                'temperature',
                shape=(),
                dtype=tf.dtypes.float32,
                initializer=tf.ones_initializer(),
                collections=temperature_collections,
                trainable=trainable)
        if data_format == 'channels_first':
            features = tf.reshape(features, [-1, height * width])
        else:
            features = tf.reshape(
                tf.transpose(features, [0, 3, 1, 2]), [-1, height * width])

        if activation is not None:
            attention = activation(features/temperature)
        else:
            attention = features/temperature

        expected_x = tf.reduce_sum(
            pos_x * attention, [1], keep_dims=True)
        expected_y = tf.reduce_sum(
            pos_y * attention, [1], keep_dims=True)
        expected_xy = tf.concatenate([expected_x, expected_y], 1)
        feature_keypoints = KTF.reshape(expected_xy, [-1, num_channels * 2])
        return feature_keypoints
