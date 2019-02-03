from __future__ import print_function
import numpy as np
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import lars
from keras.models import Sequential
from keras.layers import Dense

def test_base_lars():
  optimizers._test_optimizer(lars.LARS(0.01))
  
def test_nesterov_lars():
  optimizers._test_optimizer(lars.LARS(0.01, nesterov=True))
  
def test_skip_list():
   X = np.random.randn((15, 30))
   y = np.random.randn(3,size=(15,1))
   model = Sequential()
   model.add(Dense(32, activation='relu', input_dim=30))
   model.add(Dense(1, activation='sigmoid'))
   model.compile(loss='mse', optimizer=lars.LARS(0.01,skip_list=['bias',]))
   model.fit(X, y, epochs=3)
  
  

  
  
test_base_lars()
test_nesterov_lars()
test_skip_list()
