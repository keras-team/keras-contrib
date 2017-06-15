import numpy as np

from keras import backend as K

all_metrics = []
all_sparse_metrics = []


def validate_metric(metric):
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    output = metric(y_a, y_b)
    assert K.eval(output).shape == ()
