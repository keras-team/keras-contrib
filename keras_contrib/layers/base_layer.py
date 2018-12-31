from keras.layers import Layer
import keras
import functools


if keras.__name__ == 'keras':
    is_tf_keras = False
elif keras.__name__ == 'tensorflow.keras':
    is_tf_keras = True
else:
    raise KeyError('Cannot detect if using keras or tf.keras.')


def to_tensorshape(shape):
    if is_tf_keras:
        import tensorflow as tf
        return tf.TensorShape(shape)
    else:
        return shape


def make_return_tensorshape(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        shape = func(*args, **kwargs)
        tensorshape = to_tensorshape(shape)
        return tensorshape
    return new_func


class TfKerasCompatibleLayer(Layer):

    def __init__(self, **kwargs):

        # ensure that we can return tuples and
        # still be compatible with tf.keras.
        self.compute_output_shape = make_return_tensorshape(self.compute_output_shape)

        super(TfKerasCompatibleLayer, self).__init__(**kwargs)

    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        shape = to_tensorshape(shape)
        return super(TfKerasCompatibleLayer, self).add_weight(name,
                                                              shape,
                                                              dtype=dtype,
                                                              initializer=initializer,
                                                              regularizer=regularizer,
                                                              trainable=trainable,
                                                              constraint=constraint)
