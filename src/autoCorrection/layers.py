from keras.engine.topology import Layer
from keras import backend as K
from keras.initializers import Constant
import tensorflow as tf


class ConstantDispersionLayer(Layer):
    '''
        An identity layer which allows us to inject extra parameters
        such as dispersion to Keras models
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(shape=(1, input_shape[1]),
                                     initializer=Constant(value=25.0),
                                     trainable=False,
                                     name='theta')
        self.theta_exp = K.minimum(K.exp(self.theta), 1e12)
        super().build(input_shape)

    def call(self, x):
        return tf.identity(x)

    def compute_output_shape(self, input_shape):
        return input_shape

