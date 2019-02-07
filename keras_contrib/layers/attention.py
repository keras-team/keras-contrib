from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


class Attention(Layer):
    """
    Simple Attention layer implementation in Keras

    This implementation is based on the paper :
    Effective Approaches to Attention-based Neural Machine Translation

    The Attention Layer is a Neural Network layer that can help
    mine the necessary information from text (or other sequential
    data). The layer essentially figures out the features in the
    text data (or time series) which are important or relevant for
    the task at hand. It can automatically zoom in on local features
    or zoom out on global features required for the task. It focuses
    its "attention" on required information and achieves this using
    a simple process involving weighted sums and activations. This
    Attention layer also comes with "context". This means that the
    weights and biases of the layer can be trained to interpret the
    meaning of words correctly in the context of the sentence.

    This layer can be used on the output of any layer which has a
    3-D output (including batch_size). Note that this layer is
    generally used on the output of recurrent layers like LSTM and
    GRU in NLP applications. Nonetheless, it can also be used on the
    output of convolutional layers in Computer Vision and NLP
    applications(as it can understand what parts of an image to focus on).

    The default activation function is 'linear'. But, this layer is generally
    used with the 'tanh' activation function (recommended).

    # Example usage :
        1). NLP

           maxlen = 72
           max_features = 120000
           input_text = Input(shape=(maxlen,))

           embedding = Embedding(max_features,
                                 embed_size,
                                 weights=[embedding_matrix],
                                 trainable=False)(input_text)

           bi_gru = Bidirectional(GRU(64,
                                      return_seqeunces=True))(embedding)

           attention = Attention(activation='tanh')(bi_gru)

       2). Computer Vision

           input_image = Input(shape=(None, None, 3))

           conv_2d = Conv2D(64,
                            (3, 3),
                            activation='relu')(input_image)

           attention = Attention(activation='sigmoid')(conv_2d)

   # Arguments
       activation : Activation applied on dot products
       weight_initializer : Initializer for weights
       bias_initializer : Initializer for biases
       weight_regularizer : Regularizer for weights
       bias_regularizer : Regularizer for biases
       weight_constraint : Constraint on weights
       bias_constraint : Constraint on biases
       use_bias : Whether or not there should be bias (boolean)

   # Input shape
       3D tensor with shape:
       (batch_size, step_dim, features_dim)
       [any 3-D Tensor with the first dimension as batch_size]

   # Output shape
       2D tensor with shape:
       (batch_size, features_dim)

   # References
       - [Dynamic-Routing-Between-Capsules]
         (https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
       - [Kaggle Kernel]
         (https://www.kaggle.com/ashishpatel26/nlp-text-analytics-solution-quora)
    """

    def __init__(self,
                 activation=None,
                 weight_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 weight_regularizer=None,
                 bias_regularizer=None,
                 weight_constraint=None,
                 bias_constraint=None,
                 use_bias=True,
                 **kwargs):

        self.supports_masking = True

        self.activation = activations.get(activation)

        self.weight_initializer = initializers.get(weight_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.weight_regularizer = regularizers.get(weight_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.weight_constraint = constraints.get(weight_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.step_dim = 0
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.step_dim = input_shape[1]
        self.features_dim = input_shape[2]

        # The weight tensor
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.weight_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.weight_regularizer,
                                 constraint=self.weight_constraint)

        if self.use_bias:
            # The bias tensor
            self.b = self.add_weight((input_shape[1],),
                                     initializer=self.bias_initializer,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inputs, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        if K.backend() == 'tensorflow':
            dot_products = K.reshape(K.dot(K.reshape(inputs, (-1, features_dim)),
                                           K.reshape(self.W, (features_dim, 1))),
                                     (-1, step_dim))

        else:
            dot_products = K.dot(inputs, self.W)

        if self.use_bias:
            dot_products += self.b

        dot_products = self.activation(dot_products)
        attention_weights = K.exp(dot_products)

        if mask is not None:
            attention_weights *= K.cast(mask, K.floatx())

        attention_weights /= K.cast(K.sum(attention_weights, axis=1, keepdims=True) + K.epsilon(), \
                                    K.floatx())

        attention_weights = K.expand_dims(attention_weights)
        weighted_output = inputs * attention_weights
        return K.sum(weighted_output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.features_dim)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'weight_initializer': initializers.serialize(self.weight_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'weight_regularizer': regularizers.serialize(self.weight_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'weight_constraint': constraints.serialize(self.weight_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'use_bias': self.use_bias}

        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
