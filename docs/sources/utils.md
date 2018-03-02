<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/generic_utils.py#L16)</span>
### CustomObjectScope

```python
keras.utils.CustomObjectScope()
```

Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

Code within a `with` statement will be able to access custom objects
by name. Changes to global custom objects persist
within the enclosing `with` statement. At end of the `with` statement,
global custom objects are reverted to state
at beginning of the `with` statement.

__Example__


Consider a custom object `MyObject` (e.g. a class):

```python
with CustomObjectScope({'MyObject':MyObject}):
    layer = Dense(..., kernel_regularizer='MyObject')
    # save, load, etc. will recognize custom object by name
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/io_utils.py#L15)</span>
### HDF5Matrix

```python
keras.utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

Representation of HDF5 dataset to be used instead of a Numpy array.

__Example__


```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

Providing `start` and `end` allows use of a slice of the dataset.

Optionally, a normalizer function (or lambda) can be given. This will
be called on every slice of data retrieved.

__Arguments__

- __datapath__: string, path to a HDF5 file
- __dataset__: string, name of the HDF5 dataset in the file specified
in datapath
- __start__: int, start of desired slice of the specified dataset
- __end__: int, end of desired slice of the specified dataset
- __normalizer__: function to be called on data when retrieved

__Returns__

An array-like HDF5 dataset.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L300)</span>
### Sequence

```python
keras.utils.Sequence()
```

Base object for fitting to a sequence of data, such as a dataset.

Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
If you want to modify your dataset between epochs you may implement `on_epoch_end`.
The method `__getitem__` should return a complete batch.

__Notes__


`Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
on each sample per epoch which is not the case with generators.

__Examples__


```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```

----

### to_categorical


```python
keras.utils.to_categorical(y, num_classes=None)
```


Converts a class vector (integers) to binary class matrix.

E.g. for use with categorical_crossentropy.

__Arguments__

- __y__: class vector to be converted into a matrix
(integers from 0 to num_classes).
- __num_classes__: total number of classes.

__Returns__

A binary matrix representation of the input.

----

### normalize


```python
keras.utils.normalize(x, axis=-1, order=2)
```


Normalizes a Numpy array.

__Arguments__

- __x__: Numpy array to normalize.
- __axis__: axis along which to normalize.
- __order__: Normalization order (e.g. 2 for L2 norm).

__Returns__

A normalized copy of the array.

----

### get_file


```python
keras.utils.get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```


Downloads a file from a URL if it not already in the cache.

By default the file at the url `origin` is downloaded to the
cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
and given the filename `fname`. The final location of a file
`example.txt` would therefore be `~/.keras/datasets/example.txt`.

Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
Passing a hash will verify the file after download. The command line
programs `shasum` and `sha256sum` can compute the hash.

__Arguments__

- __fname__: Name of the file. If an absolute path `/path/to/file.txt` is
specified the file will be saved at that location.
- __origin__: Original URL of the file.
- __untar__: Deprecated in favor of 'extract'.
boolean, whether the file should be decompressed
- __md5_hash__: Deprecated in favor of 'file_hash'.
md5 hash of the file for verification
- __file_hash__: The expected hash string of the file after download.
The sha256 and md5 hash algorithms are both supported.
- __cache_subdir__: Subdirectory under the Keras cache dir where the file is
saved. If an absolute path `/path/to/folder` is
specified the file will be saved at that location.
- __hash_algorithm__: Select the hash algorithm to verify the file.
options are 'md5', 'sha256', and 'auto'.
The default 'auto' detects the hash algorithm in use.
- __extract__: True tries extracting the file as an Archive, like tar or zip.
- __archive_format__: Archive format to try for extracting the file.
Options are 'auto', 'tar', 'zip', and None.
'tar' includes tar, tar.gz, and tar.bz files.
The default 'auto' is ['tar', 'zip'].
None or an empty list will return no matches found.
- __cache_dir__: Location to store cached files, when None it
defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

__Returns__

Path to the downloaded file

----

### print_summary


```python
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=<built-in function print>)
```


Prints a summary of a model.

__Arguments__

- __model__: Keras model instance.
- __line_length__: Total length of printed lines
(e.g. set this to adapt the display to different
terminal window sizes).
- __positions__: Relative or absolute positions of log elements in each line.
If not provided, defaults to `[.33, .55, .67, 1.]`.
- __print_fn__: Print function to use.
It will be called on each line of the summary.
You can set it to a custom function
in order to capture the string summary.

----

### plot_model


```python
keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
```


Converts a Keras model to dot format and save to a file.

__Arguments__

- __model__: A Keras model instance
- __to_file__: File name of the plot image.
- __show_shapes__: whether to display shape information.
- __show_layer_names__: whether to display layer names.
- __rankdir__: `rankdir` argument passed to PyDot,
a string specifying the format of the plot:
'TB' creates a vertical plot;
'LR' creates a horizontal plot.

----

### multi_gpu_model


```python
keras.utils.multi_gpu_model(model, gpus)
```


Replicates a model on different GPUs.

Specifically, this function implements single-machine
multi-GPU data parallelism. It works in the following way:

- Divide the model's input(s) into multiple sub-batches.
- Apply a model copy on each sub-batch. Every model copy
is executed on a dedicated GPU.
- Concatenate the results (on CPU) into one big batch.

E.g. if your `batch_size` is 64 and you use `gpus=2`,
then we will divide the input into 2 sub-batches of 32 samples,
process each sub-batch on one GPU, then return the full
batch of 64 processed samples.

This induces quasi-linear speedup on up to 8 GPUs.

This function is only available with the TensorFlow backend
for the time being.

__Arguments__

- __model__: A Keras model instance. To avoid OOM errors,
this model could have been built on CPU, for instance
(see usage example below).
- __gpus__: Integer >= 2, number of on GPUs on which to create
model replicas.

__Returns__

A Keras `Model` instance which can be used just like the initial
`model` argument, but which distributes its workload on multiple GPUs.

__Example__


```python
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# Instantiate the base model (or "template" model).
# We recommend doing this with under a CPU device scope,
# so that the model's weights are hosted on CPU memory.
# Otherwise they may end up hosted on a GPU, which would
# complicate weight sharing.
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)

# Save model via the template model (which shares the same weights):
model.save('my_model.h5')
```

__On model saving__


To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
with the template model (the argument you passed to `multi_gpu_model`),
rather than the model returned by `multi_gpu_model`.
