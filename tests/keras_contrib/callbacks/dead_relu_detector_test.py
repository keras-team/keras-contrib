import pytest
import numpy as np
import sys

if sys.version_info > (3, 0):
    from io import StringIO
else:
    from StringIO import StringIO

from keras_contrib import callbacks
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Flatten, Activation
from keras import backend as K

n_out = 11
# with 1 neuron dead, 1/11 is just below the threshold of 10% with verbose = False


def check_print(do_train, expected_warnings, nr_dead=None, perc_dead=None):
    """
    Receive stdout to check if correct warning message is delivered
    :param nr_dead: int
    :param perc_dead: float, 10% should be written as 0.1
    """

    saved_stdout = sys.stdout

    out = StringIO()
    out.flush()
    sys.stdout = out    # overwrite current stdout

    do_train()

    # get prints, can be something like: "Layer
    # dense (#0) has 2 dead neurons (20.00%)!"
    stdoutput = out.getvalue().strip()
    str_to_count = "dead neurons"
    count = stdoutput.count(str_to_count)

    sys.stdout = saved_stdout   # restore stdout
    out.close()

    assert expected_warnings == count
    if expected_warnings and (nr_dead is not None):
        str_to_check = 'has {} dead'.format(nr_dead)
        assert str_to_check in stdoutput, '"{}" not in "{}"'.format(str_to_check,
                                                                    stdoutput)
    if expected_warnings and (perc_dead is not None):
        str_to_check = 'neurons ({:.2%})!'.format(perc_dead)
        assert str_to_check in stdoutput, '"{}" not in "{}"'.format(str_to_check,
                                                                    stdoutput)


def test_DeadDeadReluDetector():
    n_samples = 9

    input_shape = (n_samples, 3, 4)  # 4 input features
    shape_out = (n_samples, 3, n_out)  # 11 output features
    shape_weights = (4, n_out)

    # ignore batch size
    input_shape_dense = tuple(input_shape[1:])

    def do_test(weights, expected_warnings, verbose, nr_dead=None, perc_dead=None):

        def do_train():
            dataset = np.ones(input_shape)    # data to be fed as training
            model = Sequential()
            model.add(Dense(n_out, activation='relu', input_shape=input_shape_dense,
                            use_bias=False, weights=[weights], name='dense'))
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones(shape_out),
                batch_size=1,
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )

        check_print(do_train, expected_warnings, nr_dead, perc_dead)

    # weights that correspond to NN with 1/11 neurons dead
    weights_1_dead = np.ones(shape_weights)
    # weights that correspond to NN with 2/11 neurons dead
    weights_2_dead = np.ones(shape_weights)
    # weights that correspond to all neurons dead
    weights_all_dead = np.zeros(shape_weights)

    weights_1_dead[:, 0] = 0
    weights_2_dead[:, 0:2] = 0

    do_test(weights_1_dead, verbose=True,
            expected_warnings=1, nr_dead=1, perc_dead=1. / n_out)
    do_test(weights_1_dead, verbose=False, expected_warnings=0)
    do_test(weights_2_dead, verbose=True,
            expected_warnings=1, nr_dead=2, perc_dead=2. / n_out)
    # do_test(weights_all_dead, verbose=True, expected_warnings=1,
    # nr_dead=n_out, perc_dead=1.)


def test_DeadDeadReluDetector_bias():
    n_samples = 9

    input_shape = (n_samples, 4)  # 4 input features
    shape_weights = (4, n_out)
    shape_bias = (n_out, )
    shape_out = (n_samples, n_out)  # 11 output features

    # ignore batch size
    input_shape_dense = tuple(input_shape[1:])

    def do_test(weights, bias, expected_warnings, verbose,
                nr_dead=None, perc_dead=None):

        def do_train():
            dataset = np.ones(input_shape)  # data to be fed as training
            model = Sequential()
            model.add(Dense(n_out, activation='relu', input_shape=input_shape_dense,
                            use_bias=True, weights=[weights, bias], name='dense'))
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones(shape_out),
                batch_size=1,
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )

        check_print(do_train, expected_warnings, nr_dead, perc_dead)

    # weights that correspond to NN with 1/11 neurons dead
    weights_1_dead = np.ones(shape_weights)
    # weights that correspond to NN with 2/11 neurons dead
    weights_2_dead = np.ones(shape_weights)
    # weights that correspond to all neurons dead
    weights_all_dead = np.zeros(shape_weights)

    weights_1_dead[:, 0] = 0
    weights_2_dead[:, 0:2] = 0

    bias = np.zeros(shape_bias)

    do_test(weights_1_dead, bias, verbose=True, expected_warnings=1,
            nr_dead=1, perc_dead=1. / n_out)
    do_test(weights_1_dead, bias, verbose=False, expected_warnings=0)
    do_test(weights_2_dead, bias, verbose=True, expected_warnings=1,
            nr_dead=2, perc_dead=2. / n_out)
    # do_test(weights_all_dead, bias, verbose=True,
    # expected_warnings=1, nr_dead=n_out, perc_dead=1.)


def test_DeadDeadReluDetector_conv():
    n_samples = 9

    # (5, 5) kernel, 4 input featuremaps and 11 output featuremaps
    if K.image_data_format() == 'channels_last':
        input_shape = (n_samples, 5, 5, 4)
    else:
        input_shape = (n_samples, 4, 5, 5)

    # ignore batch size
    input_shape_conv = tuple(input_shape[1:])
    shape_weights = (5, 5, 4, n_out)
    shape_out = (n_samples, n_out)

    def do_test(weights_bias, expected_warnings, verbose,
                nr_dead=None, perc_dead=None):
        """
        :param perc_dead: as float, 10% should be written as 0.1
        """

        def do_train():
            dataset = np.ones(input_shape)  # data to be fed as training
            model = Sequential()
            model.add(Conv2D(n_out, (5, 5), activation='relu',
                             input_shape=input_shape_conv,
                             use_bias=True, weights=weights_bias, name='conv'))
            model.add(Flatten())  # to handle Theano's categorical crossentropy
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones(shape_out),
                batch_size=1,
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )

        check_print(do_train, expected_warnings, nr_dead, perc_dead)

    # weights that correspond to NN with 1/11 neurons dead
    weights_1_dead = np.ones(shape_weights)
    weights_1_dead[..., 0] = 0
    # weights that correspond to NN with 2/11 neurons dead
    weights_2_dead = np.ones(shape_weights)
    weights_2_dead[..., 0:2] = 0
    # weights that correspond to NN with all neurons dead
    weights_all_dead = np.zeros(shape_weights)

    bias = np.zeros((11, ))

    weights_bias_1_dead = [weights_1_dead, bias]
    weights_bias_2_dead = [weights_2_dead, bias]
    weights_bias_all_dead = [weights_all_dead, bias]

    do_test(weights_bias_1_dead, verbose=True, expected_warnings=1,
            nr_dead=1, perc_dead=1. / n_out)
    do_test(weights_bias_1_dead, verbose=False, expected_warnings=0)
    do_test(weights_bias_2_dead, verbose=True, expected_warnings=1,
            nr_dead=2, perc_dead=2. / n_out)
    # do_test(weights_bias_all_dead, verbose=True, expected_warnings=1,
    # nr_dead=n_out, perc_dead=1.)


def test_DeadDeadReluDetector_activation():
    """
    Tests that using "Activation" layer does not throw error
    """
    input_data = Input(shape=(1,))
    output_data = Activation('relu')(input_data)
    model = Model(input_data, output_data)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.fit(
        np.array([[1]]),
        np.array([[1]]),
        epochs=1,
        validation_data=(np.array([[1]]), np.array([[1]])),
        callbacks=[callbacks.DeadReluDetector(np.array([[1]]))]
    )


if __name__ == '__main__':
    pytest.main([__file__])
