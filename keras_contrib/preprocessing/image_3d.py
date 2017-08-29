#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import warnings
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator
import scipy.ndimage as ndi


def make_rotation_matrix(rotation_range_value, choice_number):
    """
    Make roation Matrix (x, y, z axis)
    Reference:
        https://www.tutorialspoint.com/computer_graphics/
               3d_transformation.htm
    :param rotation_range_value: setting roation degree
    :param choice_number: which axis choose
    :return: rotation matrix
    """
    if rotation_range_value != 0.0:
        theta = np.pi / 180 * rotation_range_value
    else:
        theta = 0

    if choice_number == 1:
        rotation_matrix_x = np.array([[1, 0, 0, 0],
                                      [0, np.cos(theta), -np.sin(theta), 0],
                                      [0, np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 0, 1]])
        return rotation_matrix_x
    elif choice_number == 2:
        rotation_matrix_y = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                                      [0, 1, 0, 0],
                                      [-np.sin(theta), 0, np.cos(theta), 0],
                                      [0, 0, 0, 1]])
        return rotation_matrix_y
    elif choice_number == 3:
        rotation_matrix_z = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                      [np.sin(theta), -np.cos(theta), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        return rotation_matrix_z


def make_shift_matrix(x,
                      img_row_axis,
                      img_col_axis,
                      img_depth_axis,
                      height_shift_range,
                      width_shift_range,
                      depth_shift_range):
    """Peroforms a random spatial shift matrix

    :param x: Input tensor. Must be 4D
    :param img_row_axis: Index of axis for rows in the input tensor.
    :param img_col_axis: Index of axis for cols in the input tensor.
    :param img_depth_axis: Index of axis for depth in the input tensor.
    :param height_shift_range: Height shift range, as
                               a float fraction of the height.
    :param width_shift_range: Width shift range, as
                              a float fraction of the width.
    :param depth_shift_range: Depth shift range, as
                              a float fraction of the depth.
    :return: shift matrix
    """
    tx = ty = tz = 0
    if height_shift_range != 0.0:
        tx = setting_shift_range(x, img_row_axis, height_shift_range)

    if width_shift_range != 0.0:
        ty = setting_shift_range(x, img_col_axis, width_shift_range)

    if depth_shift_range != 0.0:
        tz = setting_shift_range(x, img_depth_axis, depth_shift_range)

    translation_matrix = np.array([[1, 0, 0, tx],
                                   [0, 1, 0, ty],
                                   [0, 0, 1, tz],
                                   [0, 0, 0, 1]])
    return translation_matrix


def setting_shift_range(input_x, axis, range_value):
    """

    :param input_x:
    :param axis:
    :param range_value:
    :return:
    """
    if range_value != 0.0:
        t = range_value * input_x.shape[axis]
    else:
        t = 0
    return t


def make_shear_matrix(shear_range, shear_value):
    """Make a random spatial shear matrix
    :param shear_range: Transformation intensity range
    :param shear_value: Transformation intensity value
    :return: shear matrix
    """
    if shear_range != 0.0:
        shear_x = shear_value
        shear_y = shear_value
        shear_z = shear_value
    else:
        shear_x = 0
        shear_y = 0
        shear_z = 0

    shear_matrix = np.array([[1, shear_x, shear_x, 0],
                             [shear_y, 1, shear_y, 0],
                             [shear_z, shear_z, 1, 0],
                             [0, 0, 0, 1]
                             ])
    return shear_matrix


def make_zoom_matrix(zoom_x, zoom_y, zoom_z):
    """Make a zoom matrix

    :param zoom_x: zoom_range for x axis
    :param zoom_y: zoom_range for y axis
    :param zoom_z: zoom_range for z axis
    :return: zoom matrix
    """
    zoom_matrix = np.array([[zoom_x, 0, 0, 0],
                            [0, zoom_y, 0, 0],
                            [0, 0, zoom_z, 0],
                            [0, 0, 0, 1]])
    return zoom_matrix


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel +
                              np.random.uniform(-intensity, intensity),
                              min_x,
                              max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y],
                              [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y],
                             [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest',
                    cval=0.):
    """Apply the 3d data transformation specified by a matrix
    :param x: 3d numpy array single data
    :param transform_matrix: Numpy array specifying
                             the geometric transformation
    :param channel_axis: Index of axis for channels in the input tensor.
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    :return: The transformed version of the input
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [ndi.interpolation.affine_transform(x_channel,
                                                         final_affine_matrix,
                                                         final_offset,
                                                         order=0,
                                                         mode=fill_mode,
                                                         cval=cval)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    """
    Example:
        https://ubaa.net/shared/processing/opencv/opencv_flip.html
    Note
        https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Remember that a slicing tuple can always be constructed as obj and used
    in the x[obj] notation. Slice objects can be used in the construction
    in place of the [start:stop:step] notation. For example, x[1:10:5,::-1]
    can also be implemented as obj = (slice(1,10,5), slice(None,None,-1));
    x[obj] . This can be useful for constructing generic code that works on
    arrays of arbitrary dimension.
    :param x:
    :param axis:
    :return:
    """
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def flip_method(flip_limit_value, x,
                img_col_axis,
                img_row_axis,
                img_depth_axis,
                horizontal_flip,
                vertical_flip,
                depth_flip,
                ):
    """Apply flip for 3d data
    :param flip_limit_value:
    :param x: 3d numpy array single data
    :param img_col_axis: Index of axis for cols in the input tensor.
    :param img_row_axis: Index of axis for rows in the input tensor.
    :param img_depth_axis: Index of axis for depth in the input tensor.
    :return: The flip of the input
    """

    if horizontal_flip:
        if flip_limit_value < 0.5:
            x = flip_axis(x, img_col_axis)

    if vertical_flip:
        if flip_limit_value < 0.5:
            x = flip_axis(x, img_row_axis)

    if depth_flip:
        if flip_limit_value < 0.5:
            x = flip_axis(x, img_depth_axis)

    return x


def make_transform_matrix(rotation_matrix, translation_matrix,
                          shear_matrix, zoom_matrix, x,
                          img_row_axis, img_col_axis, img_depth_axis,
                          img_channel_axis, fill_mode, cval):
    """Apply the transformation 3d data

    :param rotation_matrix:
    :param translation_matrix:
    :param shear_matrix:
    :param zoom_matrix:
    :param x:
    :param img_row_axis:
    :param img_col_axis:
    :param img_depth_axis:
    :param img_channel_axis:
    :return:
    """
    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                            translation_matrix),
                                     shear_matrix),
                              zoom_matrix)

    h, w, d = x.shape[img_row_axis], x.shape[img_col_axis], x.shape[img_depth_axis]  # noqa

    transform_matrix = transform_matrix_offset_center(transform_matrix, h,
                                                      w, d)
    x = apply_transform(x, transform_matrix, img_channel_axis,
                        fill_mode=fill_mode, cval=cval)
    return x


def random_transform(x,
                     row_axis=1,
                     col_axis=2,
                     depth_axis=3,
                     channel_axis=4,
                     rotation_range=0.,
                     rotation_choice_number=1,
                     height_shift_range=0.,
                     width_shift_range=0.,
                     depth_shift_range=0,
                     shear_range=0.,
                     zoom_range=[1, 1, 1],
                     fill_mode='nearest',
                     cval=0.,
                     channel_shift_range=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     depth_flip=False,
                     ):
    """
    Reference:
        https://www.tutorialspoint.com/computer_graphics/
            3d_transformation.htm
    :param x:
    :return:
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = row_axis - 1
    img_col_axis = col_axis - 1
    img_depth_axis = depth_axis - 1
    img_channel_axis = channel_axis - 1
    # use composition of homographies
    # to generate final transform that needs to be applied

    rotation_range = np.random.uniform(-rotation_range, rotation_range)

    rotation_matrix = make_rotation_matrix(rotation_range,
                                           rotation_choice_number)
    height_shift_range = np.random.uniform(-height_shift_range,
                                           height_shift_range)
    width_shift_range = np.random.uniform(-width_shift_range,
                                          width_shift_range)
    depth_shift_range = np.random.uniform(-depth_shift_range,
                                          depth_shift_range)

    translation_matrix = make_shift_matrix(x,
                                           img_row_axis,
                                           img_col_axis,
                                           img_depth_axis,
                                           height_shift_range,
                                           width_shift_range,
                                           depth_shift_range)

    shear_matrix = make_shear_matrix(
        shear_range,
        np.random.uniform(-shear_range, shear_range))

    if zoom_range[0] == 1.0 and zoom_range[1] == 1.0 and \
                    zoom_range[2] == 1.0:  # noqa
        zoom_x, zoom_y, zoom_z = 1.0, 1.0, 1.0
    else:
        zoom_x, zoom_y, zoom_z = \
            np.random.uniform(zoom_range[0], zoom_range[1], 3)

    zoom_matrix = make_zoom_matrix(zoom_x, zoom_y, zoom_z)

    x = make_transform_matrix(rotation_matrix, translation_matrix,
                              shear_matrix, zoom_matrix, x,
                              img_row_axis, img_col_axis,
                              img_depth_axis,
                              img_channel_axis,
                              fill_mode,
                              cval)

    if channel_shift_range != 0:
        x = random_channel_shift(x,
                                 channel_shift_range,
                                 img_channel_axis)

    x = flip_method(np.random.random(), x,
                    img_col_axis,
                    img_row_axis,
                    img_depth_axis,
                    horizontal_flip,
                    vertical_flip,
                    depth_flip,
                    )

    return x


class Image3DDataGenerator(ImageDataGenerator):
    """Generate minibatches of image data with real-time data augmentation.
        # Arguments
            rotation_range: degrees (0 to 180).
            width_shift_range: fraction of total width.
            height_shift_range: fraction of total height.
            depth_shift_range: fraction of total depth.
            shear_range: shear intensity (shear angle in radians).
            zoom_range: amount of zoom. if scalar z, zoom will be randomly
                picked in the range [1-z, 1+z]. A sequence of two can be passed
                instead to select this range.
            channel_shift_range: shift range for each channels.
            fill_mode: points outside the boundaries are filled according to
                the given mode ('constant', 'nearest', 'reflect' or 'wrap').
                Default is 'nearest'.
            cval: value used for points outside the boundaries when fill_mode
                is 'constant'. Default is 0.
            horizontal_flip: whether to randomly flip images horizontally.
            vertical_flip: whether to randomly flip images vertically.
            depth_flip: whether to randomly flip images depthly.
            rescale: rescaling factor. If None or 0, no rescaling is applied,
                otherwise we multiply the data by the value provided
                (before applying any other transformation).
            preprocessing_function: function that will be implied on each
                input.
                The function will run before any other modification on it.
                The function should take one argument:
                one image (Numpy tensor with rank 3),
                and should output a Numpy tensor with the same shape.
            data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
                (the depth) is at index 1, in 'channels_last' mode it is at index 3.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
            Reference:
                https://github.com/stratospark/keras-multiprocess-image-data-generator  # noqa
    """
    def __init__(self,
                 rotation_range=0.,
                 rotation_choice_number=1,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 depth_shift_range=0,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 depth_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 ):
        if data_format is None:
            data_format = K.image_data_format()
        self.argumentation_parameter = {"rotation_range": rotation_range,
                                        "rotation_choice_number": rotation_choice_number,  # noqa
                                        "width_shift_range": width_shift_range,
                                        "height_shift_range": height_shift_range,  # noqa
                                        "depth_shift_range": depth_shift_range,
                                        "shear_range": shear_range,
                                        "zoom_range": zoom_range,
                                        "horizontal_flip": horizontal_flip,
                                        "vertical_flip": vertical_flip,
                                        "depth_flip": depth_flip
                                        }
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:  # noqa
            raise ValueError('data_format should be "channels_last" (channel after row and '  # noqa
                             'column) or "channels_first" (channel before row and column). '  # noqa
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
            self.depth_axis = 4
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 1
            self.col_axis = 2
            self.depth_axis = 3

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.argumentation_parameter["zoom_range"] = [1 - zoom_range, 1 + zoom_range,  # noqa
                               1 + zoom_range]
        elif len(zoom_range) == 3:
            self.argumentation_parameter["zoom_range"] = [zoom_range[0], zoom_range[1],  # noqa
                               zoom_range[2]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='npy'):

        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
        )

    def random_transform(self, x):
        return random_transform(x,
                                row_axis=self.row_axis,
                                col_axis=self.col_axis,
                                depth_axis=self.depth_axis,
                                channel_axis=self.channel_axis,
                                rotation_range=self.argumentation_parameter["rotation_range"],  # noqa
                                rotation_choice_number=self.argumentation_parameter["rotation_choice_number"],  # noqa
                                height_shift_range=self.argumentation_parameter["height_shift_range"],  # noqa
                                width_shift_range=self.argumentation_parameter["width_shift_range"],  # noqa
                                depth_shift_range=self.argumentation_parameter["depth_shift_range"],  # noqa
                                shear_range=self.argumentation_parameter["shear_range"],  # noqa
                                zoom_range=self.argumentation_parameter["zoom_range"],  # noqa
                                fill_mode=self.fill_mode,
                                cval=self.cval,
                                channel_shift_range=self.channel_shift_range,
                                horizontal_flip=self.argumentation_parameter["horizontal_flip"],  # noqa
                                vertical_flip=self.argumentation_parameter["vertical_flip"],  # noqa
                                depth_flip=self.argumentation_parameter["depth_flip"])  # noqa

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            argument: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '  # noqa
                'following the data format convention "' + self.data_format + '" '  # noqa
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '  # noqa
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '  # noqa
                'However, it was passed an array with shape ' + str(x.shape) +  # noqa
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')  # noqa

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)

        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                          dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format='default',
                 save_to_dir=None, save_prefix='', save_format='npy'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())
        if self.x.ndim != 5:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 4 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            raise ValueError('NumpyArrayIterator is set to use the '
                             'dimension ordering convention "' +
                             data_format + '" '
                             '(channels on axis ' + str(channels_axis) +
                             '), i.e. expected '
                             'either 1, 3 or 4 channels on axis ' +
                             str(channels_axis) + '. '
                             'However, it was passed an array with shape ' +
                             str(self.x.shape) +
                             ' (' + str(self.x.shape[channels_axis]) +
                             ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size,
                                                 shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = \
                next(self.index_generator)

        batch_x = np.zeros(tuple([current_batch_size] +
                                 list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))  # noqa
            batch_x[i] = x
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y
