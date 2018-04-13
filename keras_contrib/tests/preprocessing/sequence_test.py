from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import numpy as np
from numpy.testing import assert_allclose
from keras_contrib.preprocessing.sequence import TimeseriesGenerator
import pytest


def test_TimeseriesGenerator():

    print("** test 0 (float types)")

    data = np.array([[i] for i in range(50)], dtype=np.float)
    targets = np.array([[float(i)] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2,
                                   batch_size=2, shuffle=False)
    x, y = data_gen[0]

    assert np.allclose(x, np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.allclose(y, np.array([[10], [11]]))

    print("** test 1 (auto types)")

    data = np.array([[i] for i in range(50)], dtype=np.float)
    targets = np.array([[i] for i in range(50)], dtype=np.float)

    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2,
                                   batch_size=2, shuffle=False)
    x, y = data_gen[0]
    assert len(data_gen) == 20
    assert np.array_equal(x, np.array([[[0], [2], [4], [6], [8]],
                                       [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y, np.array([[10], [11]]))

    x, y = data_gen[-1]

    assert np.array_equal(x, np.array([[[38], [40], [42], [44], [46]],
                                       [[39], [41], [43], [45], [47]]]))
    assert np.array_equal(y, np.array([[48], [49]]))

    print("** test 2 (batch_size=4)")
    data_gen = TimeseriesGenerator(data, targets, length=10, batch_size=4)
    assert len(data_gen) == 10
    x, y = data_gen[0]
    assert np.array_equal(x[1], np.array(
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))
    assert np.array_equal(y, np.array([[10], [11], [12], [13]]))

    data_gen = TimeseriesGenerator(
        data, targets, length=10, reverse=True, batch_size=2)
    x, y = data_gen[0]
    assert np.array_equal(x[1, 0], np.array([10]))

    print("** test 3 (when sampling_rate is not a multiple of length)")
    data_gen = TimeseriesGenerator(
        data, targets, length=10, sampling_rate=3, batch_size=2)

    assert len(data_gen) == 10

    print("** test 4 (stateful)")
    data_gen = TimeseriesGenerator(
        data, targets, length=10, sampling_rate=2, batch_size=12, stateful=True)

    assert data_gen.stride == 2
    assert data_gen.batch_size == 10

    print("** test 5 (text sequences seq2one)")
    txt = bytearray("Keras is simple.", 'utf-8')
    data_gen = TimeseriesGenerator(txt, txt, length=10, batch_size=1)

    # for i in range(len(data_gen)):
    #    print(data_gen[i][0].tostring(), "->'%s'" % data_gen[i][1].tostring())

    assert data_gen[-1][0].shape == (1, 10) and data_gen[-1][1].shape == (1,)
    assert data_gen[-1][0].tostring() == u" is simple"
    assert data_gen[-1][1].tostring() == u"."

    print("** test 6 (text sequences seq2seq)")
    data_gen = TimeseriesGenerator(txt, txt, length=10, target_seq=True)

    assert data_gen[-1][0].shape == (1,
                                     10) and data_gen[-1][1].shape == (1, 10, 1)
    # for i in range(len(data_gen)):
    #    print(data_gen[i][0].tostring(), "->'%s'" % data_gen[i][1].tostring())

    assert data_gen[0][1].tostring() == u"eras is si"

    print("** previous tests (modified for new length semantic)")

    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2, reverse=True,
                                   batch_size=2)
    assert len(data_gen) == 20
    assert (np.allclose(data_gen[0][0],
                        np.array([[[8], [6], [4], [2], [0]],
                                  [[9], [7], [5], [3], [1]]])))
    assert (np.allclose(data_gen[0][1],
                        np.array([[10], [11]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2, shuffle=True,
                                   batch_size=1)
    batch = data_gen[0]
    r = batch[1][0][0]
    assert (np.allclose(batch[0],
                        np.array([[[r - 10],
                                   [r - 8],
                                   [r - 6],
                                   [r - 4],
                                   [r - 2]]])))
    assert (np.allclose(batch[1], np.array([[r], ])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2, stride=2,
                                   batch_size=2)
    assert len(data_gen) == 10
    assert (np.allclose(data_gen[1][0],
                        np.array([[[4], [6], [8], [10], [12]],
                                  [[6], [8], [10], [12], [14]]])))
    assert (np.allclose(data_gen[1][1],
                        np.array([[14], [16]])))

    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2)
    assert len(data_gen) == 5
    assert (np.allclose(data_gen[0][0],
                        np.array([[[10], [12], [14], [16], [18]],
                                  [[11], [13], [15], [17], [19]]])))
    assert (np.allclose(data_gen[0][1],
                        np.array([[20], [21]])))

    data = np.array([np.random.random_sample((1, 2, 3, 4)) for i in range(50)])
    targets = np.array([np.random.random_sample((3, 2, 1)) for i in range(50)])
    data_gen = TimeseriesGenerator(data, targets,
                                   length=5, sampling_rate=2,
                                   start_index=10, end_index=30,
                                   batch_size=2)

    assert len(data_gen) == 5
    assert np.allclose(data_gen[0][0], np.array(
        [np.array(data[10:19:2]), np.array(data[11:20:2])]))
    assert (np.allclose(data_gen[0][1],
                        np.array([targets[20], targets[21]])))


if __name__ == '__main__':
    test_TimeseriesGenerator()
