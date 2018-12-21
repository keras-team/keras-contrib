from keras.backend import cntk_backend as KCN


def moments(x, axes, shift=None, keep_dims=False):
    ''' Calculates and returns the mean and variance of the input '''
    mean, variant = KCN._moments(x, axes=axes, shift=shift, keep_dims=keep_dims)
    return mean, variant
