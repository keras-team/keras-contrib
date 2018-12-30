import keras.backend as K


def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Copy of the function of keras-team/keras because it's not in the public API
    So we can't use the function in keras-team/keras to test tf.keras

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def normalize_data_format(value):
    """Checks that the value correspond to a valid data format.

    Copy of the function in keras-team/keras because it's not public API.

    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```

    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format
