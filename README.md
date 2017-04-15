# keras-contrib : Keras community contributions

[![Build Status](https://travis-ci.org/farizrahman4u/keras-contrib.svg?branch=master)](https://travis-ci.org/farizrahman4u/keras-contrib)

This library is the official extension repository for the python deep learning library [Keras](http://www.keras.io). It contains additional layers, activations, loss functions, optimizers, etc. which are not yet available within Keras itself. All of these additional modules can be used in conjunction with core Keras models and modules.

As the community contributions in Keras-Contrib are tested, used, validated, and their utility proven, they may be integrated into the Keras core repository. In the interest of keeping Keras succinct, clean, and powerfully simple, only the most useful contributions make it into Keras. This contribution repository is both the proving ground for new functionality, and the archive for functionality that (while useful) may not fit well into the Keras paradigm.

---
## Installation

For instructions on how to install Keras, see https://keras.io/#installation

```shell
git clone https://www.github.com/farizrahman4u/keras-contrib.git
cd keras-contrib
python setup.py install
```

Alternatively, using pip:

```shell
sudo pip install git+https://www.github.com/farizrahman4u/keras-contrib.git
```

For contributor guidelines see [CONTRIBUTING.md](CONTRIBUTING.md)

---
## Example Usage

Modules from the Keras-Contrib library are used in the same way as modules within Keras itself.

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# I wish Keras had the Parametric Exponential Linear activation..
# Oh, wait..!
from keras_contrib.layers.advanced_activations import PELU

# Create the Keras model, including the PELU advanced activation
model = Sequential()
model.add(Dense(100, input_shape=(10,)))
model.add(PELU())

# Compile and fit on random data
model.compile(loss='mse', optimizer='adam')
model.fit(x=np.random.random((100, 10)), y=np.random.random((100, 100)), epochs=5, verbose=0)

# Save our model
model.save('example.h5')
```
```python
from keras.models import load_model
from keras_contrib.layers.advanced_activations import PELU

# Load our model
model = load_model('example.h5')
```

### A Common "Gotcha"

As Keras-Contrib is external to the Keras core, loading a model requires a bit more work. While a pure Keras model is loadable with nothing more than an import of `keras.models.load_model`, a model which contains a contributed module requires an additional import of `keras_contrib`:

```python
# Required, as usual
from keras.models import load_model

# Recommended method; requires knowledge of the underlying architecture of the model
from keras_contrib.layers.advanced_activations import PELU

# Not recommended; however this will correctly find the necessary contrib modules
from keras_contrib import *

# Load our model
model = load_model('example.h5')
```
