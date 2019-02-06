from keras import backend as K


def squash(x, axis=-1):
    """
    Squash activation function (generally used in Capsule layers).
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x
