'''
Trains a DenseNet-40-12 model on a reduced Pascal VOC segmentation dataset.

Gets a 70% accuracy score after 100 epochs.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from keras_contrib.losses.jaccard import jaccard_distance
from keras_contrib.datasets import pascal_voc
from keras_contrib.applications.densenet import DenseNetFCN

batch_size = 64
nb_classes = 21
epochs = 100

img_rows, img_cols = 32, 32
img_channels = 3

# Parameters for the DenseNet model builder
img_dim = (img_channels, img_rows,
           img_cols) if K.image_data_format() == 'channels_first' else (
               img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 16
dropout_rate = 0.0  # 0.0 for data augmentation

conf = pascal_voc.voc_config()
pascal_folder = os.path.join(conf['pascal_berkeley_root'], 'dataset')

# download dataset.. about an hour but you only need to do it once
if not os.path.isdir(pascal_folder):
    pascal_voc.data_pascal_voc.run()

# load the data using dual imagedata generators
# we create two instances with the same arguments
data_gen_args = dict()
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_generator = image_datagen.flow_from_directory(
    pascal_folder,
    target_size=(img_rows, img_cols),
    class_mode=None,
    classes=["img"],
    seed=seed)
mask_generator = image_datagen.flow_from_directory(
    pascal_folder,
    target_size=(img_rows, img_cols),
    class_mode=None,
    classes=["cls_png"],
    seed=seed)


def transform(gen_x, gen_y):
    """Combine two generatorsand one-hot transform the labels."""
    while True:
        x_batch = next(gen_x) / 255.0
        y_batch = next(gen_y)
        y_batch = to_categorical(
            y_batch[:, :, :, 0],
            num_classes=nb_classes).reshape(y_batch.shape[:3] + (nb_classes, ))
        yield x_batch, y_batch

train_generator = transform(image_generator, mask_generator)

# Create the model
model = DenseNetFCN(
    #     depth=depth,
    nb_dense_block=nb_dense_block,
    growth_rate=growth_rate,
    #     nb_filter=nb_filter,
    dropout_rate=dropout_rate,
    input_shape=img_dim,
    classes=nb_classes
)
print('Model created')
model.summary()

optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(
    loss=jaccard_distance,
    optimizer=optimizer,
    metrics=['acc', categorical_crossentropy])
print('Finished compiling')

weights_file = 'DenseNet-40-12-PASCAL-10.h5'

# setup some callbacks
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=np.sqrt(0.1),
    cooldown=0,
    patience=10,
    min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)
model_checkpoint = ModelCheckpoint(
    weights_file,
    monitor='val_acc',
    save_best_only=True,
    save_weights_only=True,
    mode='auto')
callbacks = [lr_reducer, early_stopper, model_checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=image_generator.n // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=2)

scores = model.evaluate(testX, Y_test, batch_size=batch_size)
print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])
