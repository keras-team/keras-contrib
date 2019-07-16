import keras.backend as K


def euclidean_distance_loss(y_true, y_pred):
    """
    The Euclidean distance between two points in Euclidean space.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        float type Euclidean distance between two data points.
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
