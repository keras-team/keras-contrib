'''
Trains a DenseNet-40-12 model on a reduced Pascal VOC segmentation dataset.

Gets a 70% accuracy score after 100 epochs.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os
import shutil

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras_contrib.losses.jaccard import jaccard_distance
from keras_contrib.datasets import pascal_voc
from keras_contrib.applications.densenet import DenseNetFCN

batch_size = 2
nb_classes = 21
epochs = 100
img_rows, img_cols = 256, 256
img_channels = 3

# Parameters for the DenseNet model builder
img_dim = (img_channels, img_rows,
           img_cols) if K.image_data_format() == 'channels_first' else (
               img_rows, img_cols, img_channels)

conf = pascal_voc.voc_config()
pascal_folder = os.path.join(conf['pascal_berkeley_root'], 'dataset')

# download dataset.. about an hour but you only need to do it once
if not os.path.isdir(pascal_folder):
    pascal_voc.data_pascal_voc.run()

# move validation files to diff folder
val_names = [
    l.strip()
    for l in open(
        os.path.join(conf['pascal_berkeley_root'], 'dataset', 'val.txt'))
]
val_x_folder = os.path.join(conf['pascal_berkeley_root'], 'dataset', 'img_val')
val_y_folder = os.path.join(conf['pascal_berkeley_root'], 'dataset',
                            'cls_png_val')
if not os.path.isdir(val_x_folder):
    os.makedirs(val_x_folder)
    for val_name in val_names:
        from_path = os.path.join(conf['pascal_berkeley_root'], 'dataset',
                                 'img', val_name + '.png')
        to_path = os.path.join(val_x_folder, val_name + '.jpg')
        shutil.move(from_path, to_path)
if not os.path.isdir(val_y_folder):
    os.makedirs(val_y_folder)
    for val_name in val_names:
        from_path = os.path.join(conf['pascal_berkeley_root'], 'dataset',
                                 'cls_png', val_name + '.png')
        to_path = os.path.join(val_y_folder, val_name + '.jpg')
        shutil.move(from_path, to_path)


def transform(gen_x, gen_y):
    while True:
        x_batch = next(gen_x) / 255.0
        y_batch = next(gen_y)
        y_batch = to_categorical(
            y_batch[:, :, :, 0],
            num_classes=nb_classes).reshape(y_batch.shape[:3] + (nb_classes, ))
        yield x_batch, y_batch


# load the data using dual imagedata generators
data_gen_args = dict()
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_generator = image_datagen.flow_from_directory(
    pascal_folder,
    class_mode=None,
    classes=["img"],
    batch_size=batch_size,
    seed=seed)
mask_generator = image_datagen.flow_from_directory(
    pascal_folder,
    class_mode=None,
    classes=["cls_png"],
    batch_size=batch_size,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = transform(image_generator, mask_generator)

image_generator_val = image_datagen.flow_from_directory(
    pascal_folder,
    class_mode=None,
    classes=["img_val"],
    batch_size=batch_size,
    seed=seed)

mask_generator_val = image_datagen.flow_from_directory(
    pascal_folder,
    class_mode=None,
    classes=["cls_png_val"],
    batch_size=batch_size,
    seed=seed)
# combine generators into one which yields image and masks
val_generator = transform(image_generator_val, mask_generator_val)

# # view some images if you like
# %matplotlib inline
# from mpl_toolkits.axes_grid1 import ImageGrid
# from matplotlib import pyplot as plt
# X, y = next(train_generator)
# figure = plt.figure(figsize=(10, 10))
# n=5
# grid = ImageGrid(figure, 111, (n ,2), axes_pad=0.3)
# for i in range(n):
#     xx = X[i]
#     yy = y[i]
#     grid[i*2].imshow(xx)
#     grid[i*2+1].imshow(yy.argmax(-1)*nb_classes)

# Create the model, a quick and small one
model = DenseNetFCN(
    nb_dense_block=3,
    growth_rate=10,
    nb_layers_per_block=2,
    reduction=0.0,
    dropout_rate=0.2,
    input_shape=img_dim,
    upsampling_conv=128,
    init_conv_filters=48,
    classes=nb_classes,
    upsampling_type='upsampling')
print('Model created')

model.summary()

optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(
    loss=jaccard_distance,
    optimizer=optimizer,
    metrics=['acc', categorical_crossentropy])
print('Finished compiling')
print('Model created')
model.summary()

optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(
    loss=jaccard_distance,
    optimizer=optimizer,
    metrics=['acc', categorical_crossentropy])
print('Finished compiling')

# Setup some callbacks
weights_file = 'DenseNet-40-12-PASCAL-10.h5'
csv_file = 'DenseNet-40-12-PASCAL-10.csv'
csv_logger = CSVLogger(csv_file)
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

# Train
history = model.fit_generator(
    train_generator,
    steps_per_epoch=image_generator.n // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=image_generator_val.n // batch_size,
    verbose=1)

# Get the score from the validation data
scores = model.evaluate_generator(
    val_generator, steps=image_generator_val.n // batch_size)
print('Validation loss : ', scores[0])
print('Validation accuracy : ', scores[1])
