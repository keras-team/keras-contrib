from keras import backend as K


def dice_loss(y_true, y_pred, smooth=1):
    """Dice similarity coefficient (DSC) loss.

    Essentially 1 minus the Dice similarity coefficient (DSC). Here, the Dice 
    similarity coefficient is used as a metric to evaluate the performance of 
    image segmentation by comparing spatial overlap between the true and predicted
    spaces. 

    A smoothing factor, which is by default 1, is applied to avoid dividing by 
    zeros.

    Dice loss = 1 - (2 * |X & Y|)/ (X^2 + Y^2)
              = 1 - 2 * sum(A*B) / sum(A^2 + B^2)

    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 1.

    # Returns
        The Dice coefficiet loss between the two tensors.

    # References
        - [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
           Segmentation](https://arxiv.org/pdf/1606.04797.pdf)

    # About GitHub
        If you open an issue or a pull request about Dice loss, please
        add `cc @alexbmp` to notify Seongmin Choi.

    """
    y_true_flat, y_pred_flat = K.flatten(y_true), K.flatten(y_pred)
    dice_nom = 2 * K.sum(y_true_flat * y_pred_flat)
    dice_denom = K.sum(K.square(y_true_flat) + K.square(y_pred_flat)) 
    dice_coef = (dice_nom + smooth) / (dice_denom + smooth)
    
    return 1 - dice_coef

