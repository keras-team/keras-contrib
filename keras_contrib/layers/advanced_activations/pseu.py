class PSEU(Layer):
    """Parametric Soft Exponential Unit with trainable alpha
    See: https://arxiv.org/pdf/1602.01321.pdf by Godfrey and Gashler
    Reference: https://github.com/keras-team/keras/issues/3842 (@hobson)
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_init: float. Initial value of the alpha weights.
        weights: initial alpha weights, as a list of 1 numpy array.
                 if both weights & alpha_init are provided, weights
                 overrides alpha_init
    # Example
        model = Sequential()
        model.add(Dense(10))
        model.add(PSEU())
    Soft Exponential f(α, x):
        α == 0:  x
        α  > 0:  (exp(αx)-1) / α + α
        α  < 0:  -ln(1-α(x + α)) / α
    """
    def __init__(self, alpha_init=0.1,
                 weights=None, **kwargs):
        self.supports_masking = True
        self.alpha_init = K.cast_to_floatx(alpha_init)
        self.initial_weights = weights
        super(PSEU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.alphas = K.variable(self.alpha_init * np.ones(input_shape),
                                 name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.build = True

    def call_alpha_gt0(self, x, alpha):
        return alpha + (K.exp(alpha * x) - 1.) / alpha

    def call_alpha_lt0(self, x, alpha):
        return - K.log(1 - alpha * (x + alpha)) / alpha

    def call(self, x, mask=None):
        return K.switch(self.alphas > 0, self.call_alpha_gt0,
                        K.switch(self.alphas < 0, self.call_alpha_lt0, x))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha_init': float(self.alpha_init)}
        base_config = super(ParametricSoftExp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
