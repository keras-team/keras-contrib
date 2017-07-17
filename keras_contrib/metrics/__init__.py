import keras.backend as K

def true_positives(y_true, y_pred):
    y_t = K.batch_flatten(y_true)
    y_p = K.batch_flatten(y_pred)
    return K.sum(y_t*y_p)

def true_negatives(y_true, y_pred):
    y_t = K.batch_flatten(1-y_true)
    y_p = K.batch_flatten(1-y_pred)
    return K.sum(y_t*y_p)

def ppv(y_true, y_pred):
    """
    Positive predictive value
    :param y_true:
    :param y_pred:
    :return:
    """
    return true_positives(y_true, y_pred) / (K.sum(y_true) + K.epsilon())

def npv(y_true, y_pred):
    """
    Negative predictive value
    :param y_true:
    :param y_pred:
    :return:
    """
    return true_negatives(y_true, y_pred) / (K.sum(1-y_true)+K.epsilon())
