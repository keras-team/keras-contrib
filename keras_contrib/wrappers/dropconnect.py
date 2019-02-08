from keras import backend as K
from keras.layers.wrappers import Wrapper


class DropConnect(Wrapper):
    """
    An implementation of DropConnect wrapper in Keras.
    This layer drops connections between a one layer and
    the next layer randomly with a given probability (rather
    than dropping activations as in classic Dropout).

    This wrapper can be used to drop the connections from
    any Keras layer (Dense, LSTM etc)

    #Example usage
        dense = DropConnect(Dense(10, activation='sigmoid'), prob=0.05)
        lstm = DropConnect(LSTM(20, activation='relu'), prob=0.2)

    #Arguments
        layer : Any Keras layer (instance of Layer class)
        prob : dropout rate (probability)

    #References
        https://github.com/andry9454/KerasDropconnect/blob/master/ddrop/layers.py
    """
    def __init__(self, layer, prob=0.1, **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob),
                                                 self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob),
                                               self.layer.bias)
        return self.layer.call(x)
