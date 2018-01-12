import keras
from keras import backend as K
from keras_contrib.losses.jaccard import jaccard_distance
import numpy as np

# Test and plot
y_pred = np.array([np.arange(-10, 10 + 0.1, 0.1)]).T
y_true = np.zeros(y_pred.shape)
name = 'jaccard_distance_loss'
try:
    loss = jaccard_distance_loss(
        K.variable(y_true), K.variable(y_pred)
    ).eval(session=K.get_session())
except Exception as e:
    print("error plotting", name, e)
else:
    plt.title(name)
    plt.plot(y_pred, loss)
    plt.show()

print("TYPE                 |Almost_right |half right |all_wrong")
y_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1., 0.]])
y_pred = np.array([[0, 0, 0.9, 0], [0, 0, 0.1, 0], [1, 1, 0.1, 1.]])

r = jaccard_distance(
    K.variable(y_true),
    K.variable(y_pred),
).eval(session=K.get_session())
print('jaccard_distance_loss', r)
assert r[0] < r[1]
assert r[1] < r[2]

r = keras.losses.binary_crossentropy(
    K.variable(y_true),
    K.variable(y_pred),
).eval(session=K.get_session())
print('binary_crossentropy', r)
print('binary_crossentropy_scaled', r / r.max())
assert r[0] < r[1]
assert r[1] < r[2]

"""
TYPE                 |Almost_right |half right |all_wrong
jaccard_distance_loss [ 0.09900928  0.89108944  3.75000238]
binary_crossentropy [  0.02634021   0.57564634  12.53243446]
binary_crossentropy_scaled [ 0.00210176  0.04593252  1.        ]
"""
