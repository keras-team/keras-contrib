'''
Trains a Residual-of-Residual Network (WRN-40-2) model on the CIFAR-10 Dataset.

Gets a 94.53% accuracy score after 150 epochs.
'''

import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from keras_contrib.applications import ResidualOfResidual

batch_size = 64
epochs = 150
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

model = ResidualOfResidual(depth=40, width=2, dropout_rate=0.0, weights=None)

optimizer = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
print('Finished compiling')

checkpoint = callbacks.ModelCheckpoint('weights/RoR-WRN-40-2-Weights.h5',
                                       monitor='val_acc',
                                       save_best_only=True,
                                       save_weights_only=True)
model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size,
                    epochs=epochs,
                    callbacks=[checkpoint],
                    validation_data=(testX, testY),
                    verbose=2)

scores = model.evaluate(testX, testY, batch_size)
print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])
