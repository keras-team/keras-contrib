import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc
from keras import backend as K
from keras.backend import tensorflow_backend as KTF
import numpy as np
import os
import copy
import warnings
from keras.backend.common import floatx, _EPSILON, image_dim_ordering, reset_uids
from keras.backend.tensorflow_backend import _preprocess_conv3d_input
from keras.backend.tensorflow_backend import _preprocess_conv3d_kernel
from keras.backend.tensorflow_backend import _preprocess_border_mode
from keras.backend.tensorflow_backend import _postprocess_conv3d_output
from keras.backend.tensorflow_backend import _preprocess_border_mode
from keras import callbacks as cbks
from keras import optimizers, objectives
from keras.engine.training import collect_metrics, weighted_objective
from keras import metrics as metrics_module
py_all = all


def _preprocess_deconv_output_shape(x, shape, dim_ordering):
    if dim_ordering == 'th':
        shape = (shape[0],) + tuple(shape[2:]) + (shape[1],)

    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape


def deconv3d(x, kernel, output_shape, strides=(1, 1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    '''3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of transposed 3D convolution.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    '''
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    output_shape = _preprocess_deconv_output_shape(
        x, output_shape, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    kernel = tf.transpose(kernel, (0, 1, 2, 4, 3))
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv3d_output(x, dim_ordering)


def extract_image_patches(X, ksizes, ssizes, border_mode="same", dim_ordering="tf"):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    ssizes : 2-d tuple with the strides size
    border_mode : 'same' or 'valid'
    dim_ordering : 'tf' or 'th'
    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_border_mode(border_mode)
    if dim_ordering == "th":
        X = KTF.permute_dimensions(X, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = KTF.int_shape(X)
    patches = tf.extract_image_patches(
        X, kernel, strides, [1, 1, 1, 1], padding)
    # Reshaping to fit Theano
    bs, w, h, ch = KTF.int_shape(patches)
    patches = tf.reshape(tf.transpose(tf.reshape(patches, [bs, w, h, -1, ch_i]), [0, 1, 2, 4, 3]),
                         [bs, w, h, ch_i, ksizes[0], ksizes[1]])
    if dim_ordering == "tf":
        patches = KTF.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def data_to_tfrecord(images, labels, filename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """ Save data into TFRecord """
    if not os.path.isfile(filename):
        num_examples = images.shape[0]

        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print('tfrecord already exists:' + filename)


def read_and_decode(filename, one_hot=True, n_class=None, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img.set_shape([28 * 28])
    img = tf.reshape(img, [28, 28, 1])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # img = tf.cast(img, tf.float32) * (1. / 255)

    label = tf.cast(features['label'], tf.int32)
    if one_hot and n_class:
        label = tf.one_hot(label, n_class)

    return img, label


def compile_tfrecord(train_model, optimizer, loss, out_tensor_lst, metrics=[], loss_weights=None):
    train_model.build(train_model)
    # train_model.build()

    train_model.optimizer = optimizers.get(optimizer)
    train_model.loss = loss
    train_model.loss_weights = loss_weights

    # prepare loss weights
    if loss_weights is None:
        loss_weights_list = [1. for _ in range(len(train_model.outputs))]
    elif isinstance(loss_weights, dict):
        for name in loss_weights:
            if name not in train_model.output_names:
                raise ValueError('Unknown entry in loss_weights '
                                 'dictionary: "' + name + '". '
                                 'Only expected the following keys: ' +
                                 str(train_model.output_names))
        loss_weights_list = []
        for name in train_model.output_names:
            loss_weights_list.append(loss_weights.get(name, 1.))
    elif isinstance(loss_weights, list):
        if len(loss_weights) != len(train_model.outputs):
            raise ValueError('When passing a list as loss_weights, '
                             'it should have one entry per model outputs. '
                             'The model has ' + str(len(train_model.outputs)) +
                             ' outputs, but you passed loss_weights=' +
                             str(loss_weights))
        loss_weights_list = loss_weights
    else:
        raise TypeError('Could not interpret loss_weights argument: ' +
                        str(loss_weights) +
                        ' - expected a list of dicts.')

    # prepare loss functions
    if isinstance(loss, dict):
        for name in loss:
            if name not in train_model.output_names:
                raise ValueError('Unknown entry in loss '
                                 'dictionary: "' + name + '". '
                                 'Only expected the following keys: ' +
                                 str(train_model.output_names))
        loss_functions = []
        for name in train_model.output_names:
            if name not in loss:
                raise ValueError('Output "' + name +
                                 '" missing from loss dictionary.')
            loss_functions.append(objectives.get(loss[name]))
    elif isinstance(loss, list):
        if len(loss) != len(train_model.outputs):
            raise ValueError('When passing a list as loss, '
                             'it should have one entry per model outputs. '
                             'The model has ' + str(len(train_model.outputs)) +
                             ' outputs, but you passed loss=' +
                             str(loss))
        loss_functions = [objectives.get(l) for l in loss]
    else:
        loss_function = objectives.get(loss)
        loss_functions = [loss_function for _ in range(
            len(train_model.outputs))]
    train_model.loss_functions = loss_functions
    weighted_losses = [weighted_objective(fn) for fn in loss_functions]

    # prepare metrics
    train_model.metrics = metrics
    train_model.metrics_names = ['loss']
    train_model.metrics_tensors = []

    # compute total loss
    total_loss = None
    for i in range(len(train_model.outputs)):
        y_true = out_tensor_lst[i]
        y_pred = train_model.outputs[i]
        _loss = loss_functions[i]
        # _loss = weighted_losses[i]
        loss_weight = loss_weights_list[i]
        # output_loss = _loss(y_true, y_pred, None, None)
        output_loss = K.mean(_loss(y_true, y_pred))
        if len(train_model.outputs) > 1:
            train_model.metrics_tensors.append(output_loss)
            train_model.metrics_names.append(
                train_model.output_names[i] + '_loss')
        if total_loss is None:
            total_loss = loss_weight * output_loss
        else:
            total_loss += loss_weight * output_loss

    # add regularization penalties
    # and other layer-specific losses
    for loss_tensor in train_model.losses:
        total_loss += loss_tensor

    # list of same size as output_names.
    # contains tuples (metrics for output, names of metrics)
    nested_metrics = collect_metrics(metrics, train_model.output_names)

    def append_metric(layer_num, metric_name, metric_tensor):
        """Helper function, used in loop below"""
        if len(train_model.output_names) > 1:
            metric_name = train_model.output_layers[
                layer_num].name + '_' + metric_name

        train_model.metrics_names.append(metric_name)
        train_model.metrics_tensors.append(metric_tensor)

    for i in range(len(train_model.outputs)):
        y_true = out_tensor_lst[i]
        y_pred = train_model.outputs[i]
        output_metrics = nested_metrics[i]

        for metric in output_metrics:
            if metric == 'accuracy' or metric == 'acc':
                # custom handling of accuracy
                # (because of class mode duality)
                output_shape = train_model.internal_output_shapes[i]
                acc_fn = None
                if output_shape[-1] == 1 or train_model.loss_functions[i] == objectives.binary_crossentropy:
                    # case: binary accuracy
                    acc_fn = metrics_module.binary_accuracy
                elif train_model.loss_functions[i] == objectives.sparse_categorical_crossentropy:
                    # case: categorical accuracy with sparse targets
                    acc_fn = metrics_module.sparse_categorical_accuracy
                else:
                    acc_fn = metrics_module.categorical_accuracy

                append_metric(i, 'acc', acc_fn(y_true, y_pred))
            else:
                metric_fn = metrics_module.get(metric)
                metric_result = metric_fn(y_true, y_pred)

                if not isinstance(metric_result, dict):
                    metric_result = {
                        metric_fn.__name__: metric_result
                    }

                for name, tensor in six.iteritems(metric_result):
                    append_metric(i, name, tensor)

    # prepare gradient updates and state updates
    train_model.optimizer = optimizers.get(optimizer)
    train_model.total_loss = total_loss

    train_model.train_function = None
    train_model.test_function = None
    train_model.predict_function = None

    # collected trainable weights and sort them deterministically.
    trainable_weights = train_model.trainable_weights
    # Sort weights by name
    trainable_weights.sort(key=lambda x: x.name)
    train_model._collected_trainable_weights = trainable_weights


def fit_tfrecord(train_model, nb_train_sample, batch_size, nb_epoch=10, verbose=1, callbacks=[],
                 initial_epoch=0):

    def _make_train_function(model):
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if model.train_function is None:
            inputs = [K.learning_phase()]

            training_updates = model.optimizer.get_updates(model._collected_trainable_weights,
                                                           model.constraints,
                                                           model.total_loss)
            updates = model.updates + training_updates

            # returns loss and metrics. Updates weights at each call.
            model.train_function = K.function(inputs,
                                              [model.total_loss] +
                                              model.metrics_tensors,
                                              updates=updates)

    ins = [1.]

    _make_train_function(train_model)
    f = train_model.train_function

    # prepare display labels
    out_labels = train_model.metrics_names

    # rename duplicated metrics name
    # (can happen with an output layer shared among multiple dataflows)
    deduped_out_labels = []
    for i, label in enumerate(out_labels):
        new_label = label
        if out_labels.count(label) > 1:
            dup_idx = out_labels[:i].count(label)
            new_label += '_' + str(dup_idx + 1)
        deduped_out_labels.append(new_label)
    out_labels = deduped_out_labels

    callback_metrics = copy.copy(out_labels)

    train_model.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks) + [train_model.history]
    if verbose:
        callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)
    out_labels = out_labels or []

    callback_model = train_model

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'nb_epoch': nb_epoch,
        'nb_sample': nb_train_sample,
        'verbose': verbose,
        'do_validation': False,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False

    sess = K.get_session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(initial_epoch, nb_epoch):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index in range(0, nb_train_sample / batch_size):
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_size
            callbacks.on_batch_begin(batch_index, batch_logs)
            outs = f(ins)
            if not isinstance(outs, list):
                outs = [outs]
            for l, o in zip(out_labels, outs):
                batch_logs[l] = o

            callbacks.on_batch_end(batch_index, batch_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()

    coord.request_stop()
    coord.join(threads)
    # sess.close()

    return train_model.history
