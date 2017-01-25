# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations
from .. import initializations
from .. import regularizers
from keras.engine import Layer
from keras.engine import InputSpec

from keras.layers.recurrent import time_distributed_dense
