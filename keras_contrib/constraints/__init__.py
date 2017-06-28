from __future__ import absolute_import
from keras.constraints import *

from .clip import Clip
from .curvature import CurvatureConstraint

# Aliases.

clip = Clip
curvature = CurvatureConstraint