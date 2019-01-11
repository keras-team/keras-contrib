'''
Trains a WRN-28-8 model on the CIFAR-10 Dataset.

Performance is slightly less than the paper, since
they use WRN-28-10 model (95.83%).

Gets a 95.54% accuracy score after 300 epochs.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator

from keras_contrib.applications.wide_resnet import WideResidualNetwork

batch_size = 64
epochs = 300
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

# We will be training the model, therefore no need to load weights
model = WideResidualNetwork(depth=28, width=8, dropout_rate=0.0, weights=None)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print('Finished compiling')
model_checkpoint = callbacks.ModelCheckpoint('WRN-28-8 Weights.h5',
                                             monitor='val_acc',
                                             save_best_only=True,
                                             save_weights_only=True)
model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size,
                    epochs=epochs,
                    callbacks=[model_checkpoint],
                    validation_data=(testX, testY))

scores = model.evaluate(testX, testY, batch_size)
print('Test loss : %0.5f' % (scores[0]))
print('Test accuracy = %0.5f' % (scores[1]))
