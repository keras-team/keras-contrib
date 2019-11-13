<h1 id="keras_contrib.losses">keras_contrib.losses</h1>


<h2 id="keras_contrib.losses.DSSIMObjective">DSSIMObjective</h2>

```python
DSSIMObjective(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0)
```
Difference of Structural Similarity (DSSIM loss function).
Clipped between 0 and 0.5

Note : You should add a regularization term like a l2 loss in addition to this one.
Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
       not be the `kernel_size` for an output of 32.

__Arguments__

- __k1__: Parameter of the SSIM (default 0.01)
- __k2__: Parameter of the SSIM (default 0.03)
- __kernel_size__: Size of the sliding window (default 3)
- __max_value__: Max value of the output (default 1.0)

<h2 id="keras_contrib.losses.jaccard_distance">jaccard_distance</h2>

```python
jaccard_distance(y_true, y_pred, smooth=100)
```
Jaccard distance for semantic segmentation.

Also known as the intersection-over-union loss.

This loss is useful when you have unbalanced numbers of pixels within an image
because it gives all classes equal weight. However, it is not the defacto
standard for image segmentation.

For example, assume you are trying to predict if
each pixel is cat, dog, or background.
You have 80% background pixels, 10% dog, and 10% cat.
If the model predicts 100% background
should it be be 80% right (as with categorical cross entropy)
or 30% (with this loss)?

The loss has been modified to have a smooth gradient as it converges on zero.
This has been shifted so it converges on 0 and is smoothed to avoid exploding
or disappearing gradient.

Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
        = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

__Arguments__

- __y_true__: The ground truth tensor.
- __y_pred__: The predicted tensor
- __smooth__: Smoothing factor. Default is 100.

__Returns__

    The Jaccard distance between the two tensors.

__References__

    - [What is a good evaluation measure for semantic segmentation?](
       http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)


<h2 id="keras_contrib.losses.crf_loss">crf_loss</h2>

```python
crf_loss(y_true, y_pred)
```
General CRF loss function depending on the learning mode.

__Arguments__

- __y_true__: tensor with true targets.
- __y_pred__: tensor with predicted targets.

__Returns__

    If the CRF layer is being trained in the join mode, returns the negative
    log-likelihood. Otherwise returns the categorical crossentropy implemented
    by the underlying Keras backend.

__About GitHub__

    If you open an issue or a pull request about CRF, please
    add `cc @lzfelix` to notify Luiz Felix.

<h2 id="keras_contrib.losses.crf_nll">crf_nll</h2>

```python
crf_nll(y_true, y_pred)
```
The negative log-likelihood for linear chain Conditional Random Field (CRF).

This loss function is only used when the `layers.CRF` layer
is trained in the "join" mode.

__Arguments__

- __y_true__: tensor with true targets.
- __y_pred__: tensor with predicted targets.

__Returns__

    A scalar representing corresponding to the negative log-likelihood.

__Raises__

- `TypeError`: If CRF is not the last layer.

__About GitHub__

    If you open an issue or a pull request about CRF, please
    add `cc @lzfelix` to notify Luiz Felix.

<h2 id="keras_contrib.losses.dice_loss">dice_loss</h2>

```python
dice_loss(y_true, y_pred, smooth=1)
```
Dice similarity coefficient (DSC) loss.

Essentially 1 minus the Dice similarity coefficient (DSC). Here, the Dice
similarity coefficient is used as a metric to evaluate the performance of
image segmentation by comparing spatial overlap between the true and predicted
spaces.

A smoothing factor, which is by default 1, is applied to avoid dividing by
zeros.

Dice loss = 1 - (2 * |X & Y|)/ (X^2 + Y^2)
          = 1 - 2 * sum(A*B) / sum(A^2 + B^2)

__Arguments__

- __y_true__: The ground truth tensor.
- __y_pred__: The predicted tensor
- __smooth__: Smoothing factor. Default is 1.

__Returns__

    The Dice coefficiet loss between the two tensors.

__References__

    - [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
       Segmentation](https://arxiv.org/pdf/1606.04797.pdf)

__About GitHub__

    If you open an issue or a pull request about Dice loss, please
    add `cc @alexbmp` to notify Seongmin Choi.
