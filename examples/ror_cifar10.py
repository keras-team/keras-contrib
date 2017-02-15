'''
Trains a Residual-of-Residual Network (WRN-40-2) model on the CIFAR-10 Dataset.

Gets a 94.53% accuracy score after 150 epochs.
'''
import numpy as np
import sklearn.metrics as metrics

import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from keras_contrib.applications.ror import ResidualOfResidual

batch_size = 64
nb_epoch = 150
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX /= 255
testX /= 255

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32)

generator.fit(trainX, seed=0)

model = ResidualOfResidual(depth=40, width=2, dropout_rate=0.0, weights='None')

optimizer = Adam(lr=1e-3)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
print("Finished compiling")

model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX),
                    nb_epoch=nb_epoch,
                    callbacks=[callbacks.ModelCheckpoint("weights/RoR-WRN-40-2-Weights.h5", monitor="val_acc",
                                                         save_best_only=True, save_weights_only=True)],
                    validation_data=(testX, testY),
                    nb_val_samples=testX.shape[0], verbose=2)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = tempY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
