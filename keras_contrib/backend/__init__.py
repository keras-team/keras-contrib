from keras import backend as K

 # We import all keras backend functions here,
 # so that files in this repo can import both
 # core and contrib backend functions with a
 # single import statement.
 
from keras.backend import *


if K.backend() == 'theano':
	from .theano_backend import *
elif K.backend() == 'tensorflow':
	from .tensorflow_backend import *
