# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import keras

import copy
import inspect
import types as python_types
import warnings

from .. import backend as K
from .. import activations
from .. import initializations
from .. import regularizers
from .. import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras.engine import Merge
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import get_from_module
