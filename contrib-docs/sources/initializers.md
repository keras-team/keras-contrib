
## Available initializers

The following built-in initializers are available as part of the `keras.initializers` module:

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py#L8)</span>
### ConvolutionAware

```python
keras_contrib.initializers.convaware.ConvolutionAware(eps_std=0.05, seed=None)
```


Initializer that generates orthogonal convolution filters in the fourier
space. If this initializer is passed a shape that is not 3D or 4D,
orthogonal initialization will be used.
__Arguments__

eps_std: Standard deviation for the random normal noise used to break
symmetry in the inverse fourier transform.
seed: A Python integer. Used to seed the random generator.
__References__

Armen Aghajanyan, https://arxiv.org/abs/1702.06295

