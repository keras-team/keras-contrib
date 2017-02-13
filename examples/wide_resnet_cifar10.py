from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator

from keras_contrib.applications.wide_resnet import WideResidualNetwork

batch_size = 64
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(featurewise_center=True,
                               featurewise_std_normalization=True,
                               rotation_range=10,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

test_generator = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True, )

test_generator.fit(testX, seed=0, augment=True)

# We will be training the model, therefore no need to load weights
model = WideResidualNetwork(depth=28, width=8, dropout_rate=0.0, weights=None)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("Finished compiling")

model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX),
                    nb_epoch=nb_epoch,
                    callbacks=[callbacks.ModelCheckpoint("WRN-28-8 Weights.h5", monitor="val_acc", save_best_only=True)],
                    validation_data=test_generator.flow(testX, testY, batch_size=batch_size),
                    nb_val_samples=testX.shape[0], )

scores = model.evaluate_generator(test_generator.flow(testX, testY, nb_epoch), testX.shape[0])
print("Accuracy = %f" % (100 * scores[1]))
