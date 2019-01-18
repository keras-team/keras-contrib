from keras.layers import Layer
import keras
import functools


if keras.__name__ == 'keras':
    is_tf_keras = False
elif keras.__name__ == 'tensorflow.keras':
    is_tf_keras = True
else:
    raise KeyError('Cannot detect if using keras or tf.keras.')


def from_tensorshape(shape):
    if is_tf_keras:
        import tensorflow as tf
        return tuple(tf.TensorShape(shape).as_list())
    else:
        return shape


def make_manipulate_tuple(func):
    @functools.wraps(func)
    def new_func(input_shape):
        func(from_tensorshape(input_shape))
    return new_func


class TfKerasCompatibleLayer(Layer):

    def __init__(self, **kwargs):

        # ensure that we can return tuples and
        # still be compatible with tf.keras.
        self.build = make_manipulate_tuple(self.build)

        super(TfKerasCompatibleLayer, self).__init__(**kwargs)

    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        return super(TfKerasCompatibleLayer, self).add_weight(name,
                                                              shape,
                                                              dtype=dtype,
                                                              initializer=initializer,
                                                              regularizer=regularizer,
                                                              trainable=trainable,
                                                              constraint=constraint)
