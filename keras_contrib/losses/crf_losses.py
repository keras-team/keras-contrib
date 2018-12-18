from .. import backend as K
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy


def crf_nll(y_true, y_pred):
    """The negative log-likelihood used to train the linear chain Conditional Random Field (CRF).

    This loss function is only used when the `layers.CRF` layer is trained in the "join" mode.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        A scalar representing corresponding to the negative log-likelihood.
    """

    crf, idx = y_pred._keras_history[:2]
    assert not crf._outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):
    """General CRF loss function depending on the learning mode.

    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.

    # Returns
        If the CRF layer is being trained in the join mode, returns the negative
        log-likelihood. Otherwise returns the categorical crossentropy implemented
        by the underlying Keras backend.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)
