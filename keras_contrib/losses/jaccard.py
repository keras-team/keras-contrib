from keras import backend as K

def jaccard_distance(y_true, y_pred, smooth=100):
    """    
    Jaccard distance is a usefull loss when you have unbalanced classes within a sample
    because it gives all classes equal weight. 
    
    For example, you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
     
    The loss has been modified to have a smooth gradient as it converges on zero.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
