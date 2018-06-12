from .. import backend as K
from keras.objectives import categorical_crossentropy
from keras.objectives import sparse_categorical_crossentropy


def crf_nll(y_true, y_pred):
    '''Usual Linear Chain CRF negative log likelihood. Used for CRF "join" mode. '''
    crf, idx = y_pred._keras_history[:2]
    assert not crf._outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):
    '''General CRF loss function, depending on the learning mode.'''
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)
