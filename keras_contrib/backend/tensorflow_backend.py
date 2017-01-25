import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc
from keras import backend as K
import numpy as np
import os
import warnings
from keras.backend.common import floatx, _EPSILON, image_dim_ordering, reset_uids
py_all = all
