# -*- coding: utf-8 -*-
"""Utilities for preprocessing sequence data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from six.moves import range

from keras.utils.data_utils import Sequence




class TimeseriesGenerator(Sequence):
    """Utility class for generating batches of temporal data.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be convertible into a 1D numpy array,
            if 2D or more, axis 0 is expected to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have at least the same length as `data`.
        length: Efective length of the outputsub-sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences.
        gap: prediction gap, i.e. numer of timesteps ahead (usually same as samplig_rate)            
            `x=data[i - (length-1)*sampling_rate - gap:i-gap+1:sampling_rate]` and `y=targets[i]`
            are used respectively as sample sequence `x` and target value `y`.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index, end_index: Data points earlier than `start_index`
            or later than `end_index` will not be used in the output sequences.
            This is useful to reserve part of the data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `True`, timesteps in each output sample will be
            in reverse chronological order.
        target_seq: Boolean: if 'True', produces full shifted sequences targets:
            If target_seq is set, for sampling rate `r`, timesteps
            `data[i - (length-1)*r - gap]`, ..., `data[i-r-gap]`, `data[i-gap]` and
            `targets[i - (length-1)*r]`, ..., `data[i-r]`, `data[i]`
            are used respectively as sample sequence and target sequence.

        
        batch_size: Number of timeseries samples in each batch
        dtype: force sample/target dtype (default is None)
        stateful: helper to set parameters for stateful learning
                
        
    # Returns
        A [Sequence](/utils/#sequence) instance of tuples (x,y)
        where x is a numpy array of shape (batch_size, length, ...)
        and y is a numpy array of shape (batch_size, ...) if target_seq is `False`
        or (batch_size, length, ...) if target_seq is `True`.
        If not specified, output dtype is infered from data dtype.
        
    # Examples
    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])
    
    
    print "** test 1"
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

    print "** test 2"
    data_gen = TimeseriesGenerator(data, targets, length=10, batch_size=4)
    assert len(data_gen) == 10
    x, y = data_gen[0]
    assert np.array_equal(x[1], np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))
    assert np.array_equal(y, np.array([[10], [11], [12], [13]]))

    data_gen = TimeseriesGenerator(data, targets, length=10, reverse=True, batch_size=2)
    x, y = data_gen[0]
    assert np.array_equal(x[1,0], np.array([10]))

    print "** test 3"
    data_gen = TimeseriesGenerator(data, targets, length=10, sampling_rate=3, batch_size=2)

    assert len(data_gen) == 10

    print "** test 4 (stateful)"
    data_gen = TimeseriesGenerator(data, targets, length=10, sampling_rate=2, batch_size=12, stateful=True)

    assert data_gen.stride == 2
    assert data_gen.batch_size == 10

    print "** test 5 (text sequences seq2one)"
    txt = bytearray("Keras is simple.")
    data_gen = TimeseriesGenerator(txt, txt, length=10, batch_size=1)

    for i in range(len(data_gen)):
        print data_gen[i][0].tostring(), " -> '%s'" % data_gen[i][1].tostring()

    assert data_gen[-1][0].shape == (1, 10) and data_gen[-1][1].shape==(1,)
    assert data_gen[-1][0].tostring() == u" is simple"
    assert data_gen[-1][1].tostring() == u"."

    print "** test 6 (text sequences seq2seq)"
    data_gen = TimeseriesGenerator(txt, txt, length=10, target_seq=True)

    assert data_gen[-1][0].shape == (1, 10) and data_gen[-1][1].shape==(1,10,1)
    for i in range(len(data_gen)):
        print data_gen[i][0].tostring(), " -> '%s'" % data_gen[i][1].tostring()

    assert data_gen[0][1].tostring() == u"eras is si"
    

    ```
    """

    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 gap = None,
                 start_index = 0,
                 end_index = None,
                 shuffle = False,
                 reverse = False,
                 target_seq = False,
                 batch_size = 1,
                 dtype = None,
                 stateful = False):
        
        assert length > 0
        assert sampling_rate > 0
        assert batch_size > 0
        assert len(data) <= len(targets)
        
        self.data = np.asarray(data)
        self.targets = np.asarray(targets)
        
        # FIXME: seems required by sparse losses on seq output
        if target_seq and len(self.targets.shape)<2:
            self.targets = np.expand_dims(targets,axis=-1)
        
        if dtype is None:
            self.data_type = self.data.dtype
            self.targets_type = self.targets.dtype
        else:
            self.data_type = dtype
            self.targets_type = dtype
            
        
        # force stateful-compatible parameters
        if stateful:
            shuffle=False
            gap = sampling_rate
            b = batch_size
            while length % b  > 0:
                b -= 1
            batch_size = b
            stride = (length // batch_size) * sampling_rate
        
        self.length = length
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        
        assert stride > 0
        self.stride = stride
        if gap is None:
            gap = sampling_rate
        self.gap = gap
        
        self.win_size = (length-1)*sampling_rate + gap
        self.start_index = start_index + self.win_size
        if end_index is None:
            end_index = np.asarray(data).shape[0]
        assert end_index<=np.asarray(data).shape[0]
        self.end_index = end_index
        self.reverse = reverse
        self.target_seq  = target_seq
        
        assert self.start_index < self.end_index
        assert self.batch_size * self.stride > 0
        assert self.batch_size * self.stride < self.end_index - self.start_index
        self.len = (self.end_index - self.start_index) // (self.batch_size * self.stride)
        assert self.len > 0
                
        self.perm = np.arange(self.start_index, self.end_index)
        if shuffle:
            np.random.shuffle(self.perm)
            
        

    def __len__(self):
        return self.len

    def _empty_batch(self, num_rows):
        samples_shape = [num_rows, self.length]
        samples_shape.extend(self.data.shape[1:])
        if self.target_seq:
            targets_shape = [num_rows, self.length]
        else:
            targets_shape = [num_rows]
        targets_shape.extend(self.targets.shape[1:])
        
        return np.empty(samples_shape, dtype=self.data_type), np.empty(targets_shape, dtype=self.targets_type)

    def __getitem__(self, index):
        while index<0:
            index += self.len
        assert index<self.len
        
        i = self.batch_size * self.stride * index
        assert i + self.batch_size * self.stride <= self.end_index
        rows = np.arange(i, i + self.batch_size * self.stride, self.stride)
        
        rows = self.perm[rows]

        samples, targets = self._empty_batch(len(rows))
        for j, row in enumerate(rows):
            indices = range(rows[j] - self.gap - (self.length - 1) * self.sampling_rate, 
                            rows[j] - self.gap + 1, self.sampling_rate)
            samples[j] = self.data[indices]
            if self.target_seq:
                shifted_indices = range(rows[j] - (self.length - 1) * self.sampling_rate, 
                            rows[j] + 1, self.sampling_rate)
                targets[j] = self.targets[shifted_indices]
            else:
                targets[j] = self.targets[rows[j]]
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets
