from .dssim import DSSIMObjective
import keras.backend as K

def dice(y_true, y_pred):
    """
    A simple DICE implementation without any weighting
    :param y_true:
    :param y_pred:
    :return:
    """
    y_t = K.batch_flatten(y_true)
    y_p = K.batch_flatten(y_pred)
    return 2.0*K.sum(y_t*y_p) / (K.sum(y_t)+K.sum(y_p)+K.epsilon())

def dice_loss(y_true, y_pred):
    """
    A simple inverted dice to use as a loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1-dice(y_true, y_pred)
