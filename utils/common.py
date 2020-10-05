import tensorflow as tf

class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units=32, use_bias=True, std_init=0.017, activation=None, **kwargs):
        """
        Independent Gaussian noise
        """
        super().__init__(self, **kwargs)
        self.units = units
        self.std_init = std_init
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        p = tf.math.sqrt(3. / input_shape[-1]).numpy()
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer=tf.initializers.RandomUniform(minval=-p, maxval=p),
                                    trainable=True)
        self.w_sigma = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer=tf.initializers.Constant(self.std_init),
                                       trainable=True)  
        self.w_epsilon = self.add_weight(shape=(input_shape[-1], self.units),
                                         initializer=tf.initializers.Constant(p),
                                         trainable=False)
        if self.use_bias:
            self.b_mu = self.add_weight(shape=(self.units,),
                                        initializer=tf.initializers.RandomUniform(minval=-p, maxval=p),
                                        trainable=True)
            self.b_sigma = self.add_weight(shape=(self.units,),
                                           initializer=tf.initializers.Constant(self.std_init),
                                           trainable=True)                           
            self.b_epsilon = self.add_weight(shape=(self.units,),
                                             initializer=tf.initializers.Constant(p),
                                             trainable=False)
    
    def call(self, inputs):
        weight_epsilon = tf.random.normal(shape=self.w_epsilon.shape)
        w = self.w_mu + self.w_sigma * weight_epsilon
        r = tf.matmul(inputs, w)
        if self.use_bias:
            bias_epsilon = tf.random.normal(shape=self.b_epsilon.shape)
            b = self.b_mu + self.b_sigma * bias_epsilon
            r += b
        if self.activation:
            return self.activation(r)
        return r

# TODO Factorised Gaussian noise
