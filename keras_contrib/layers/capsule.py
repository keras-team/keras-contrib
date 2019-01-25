# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial

from keras import backend as K
from keras_contrib import backend as KC
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras.utils import get_custom_objects
import numpy as np

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

class Capsule(Layer):
     """Capsule Layer implementation in Keras
       
       The Capsule Layer is a Neural Network Layer which helps modeling relationships in image and sequential data better.
       It achieves this by understanding the spatial relationships between objects (in images) or words (in text) by encoding
       additional information about the image or text, such as angle of rotation, thickness and brightness, relative proportions etc.
       This layer can be used instead of pooling layers to lower dimensions and still capture important information about the 
       relationships and structures within the data. A normal pooling layer would lose a lot of this information.
       
       This layer can be used on the output of any layer which has a 3-D output (including batch_size). For example, in 
       image classification, it can be used on the output of a Conv2D layer for Computer Vision applications. Also, it can be 
       used on the output of a GRU or LSTM Layer (Bidirectional or Unidirectional) for NLP applications
       
       # Example usage :
       
           1). Computer Vision
           input_image = Input(shape=(None, None, 3))
           conv_2d = Conv2D(64, (3, 3), activation='relu')(input_image)
           capsule = Capsule(num_capsule=10, dim_capsule=16, routings=3, share_weights=True)(conv_2d)
       
           2). NLP
           maxlen = 72
           max_features = 120000
           input_text = Input(shape=(maxlen,))
           embedding = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(input_text)
           bi_gru = Bidirectional(GRU(64, return_seqeunces=True))(embedding)
           capsule = Capsule(num_capsule=5, dim_capsule=5, routings=4, share_weights=True)(bi_gru)
           
       # Arguments
           num_capsule : Number of Capsules (int)
           dim_capsules : Dimensions of the vector output of each Capsule (int)
           routings : Number of dynamic routings in the Capsule Layer (int)
           share_weights : Whether to share weights between Capsules or not (boolean)
           
       # Input shape
            3D tensor with shape:
            (batch_size, input_num_capsule, input_dim_capsule) [any 3-D Tensor with the first dimension as batch_size]
        
       # Output shape
            3D tensor with shape:
            (batch_size, num_capsule, dim_capsule)
       # References
        - [Dynamic-Routing-Between-Capsules](https://arxiv.org/pdf/1710.09829.pdf)
        - [Keras-Examples-CIFAR10-CNN-Capsule](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn_capsule.py)"""
        
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = KC.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = KC.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, axis=1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    
    def get_config(self):
        config = {'num_capsule': self.num_capsule,
                  'dim_capsule': self.dim_capsule,
                  'routings': self.routings,
                  'share_weights': self.share_weights, 
                  'activation': activations.serialize(self.activation)}
        
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
 
get_custom_objects().update({'Capsule': Capsule})


