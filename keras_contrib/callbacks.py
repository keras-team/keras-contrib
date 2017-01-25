from __future__ import absolute_import
from __future__ import print_function

import os
import csv

import numpy as np
import time
import json
import warnings

from collections import deque
from collections import OrderedDict
from collections import Iterable
from keras.utils.generic_utils import Progbar
from keras import backend as K
from pkg_resources import parse_version

try:
    import requests
except ImportError:
    requests = None
