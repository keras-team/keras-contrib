import pytest
import numpy as np

from keras import backend as K
from keras_contrib import constraints


test_values = [0.1, 0.5, 3, 8, 1e-7]
np.random.seed(3537)
example_array = np.random.random((100, 100)) * 100. - 50.
example_array[0, 0] = 0.  # 0 could possibly cause trouble


def test_clip():
    clip_instance = constraints.clip()
    clipped = clip_instance(K.variable(example_array))
    assert(np.max(np.abs(K.eval(clipped))) <= K.cast_to_floatx(0.01))
    clip_instance = constraints.clip(0.1)
    clipped = clip_instance(K.variable(example_array))
    assert(np.max(np.abs(K.eval(clipped))) <= K.cast_to_floatx(0.1))

if __name__ == '__main__':
    pytest.main([__file__])
