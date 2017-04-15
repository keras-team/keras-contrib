import pytest
import numpy as np

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib import metrics


all_metrics = []
all_sparse_metrics = []


def test_metrics():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for metric in all_metrics:
        output = metric(y_a, y_b)
        assert K.eval(output).shape == ()


if __name__ == '__main__':
    pytest.main([__file__])
