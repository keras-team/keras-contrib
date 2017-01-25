import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib import constraints


test_values = [0.1, 0.5, 3, 8, 1e-7]
np.random.seed(3537)
example_array = np.random.random((100, 100)) * 100. - 50.
example_array[0, 0] = 0.  # 0 could possibly cause trouble


if __name__ == '__main__':
    pytest.main([__file__])
