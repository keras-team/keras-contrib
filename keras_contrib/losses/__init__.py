from __future__ import absolute_import
from . import segmentation_losses
from . import dssim

# Globally-importable losses
from .segmentation_losses import binary_crossentropy

from .dssim import DSSIMObjective
