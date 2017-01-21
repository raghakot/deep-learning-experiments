from keras.layers.convolutional import Convolution2D
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf


class Convolution2DNew(Convolution2D):
    def call(self, x, mask=None):
        # 1e-8 is used to prevent division by 0
        W_norm = self.W / (tf.sqrt(tf.reduce_sum(tf.square(self.W), axis=[0, 1, 2], keep_dims=True)) + 1e-8)

        output = K.conv2d(x, W_norm, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output


class DenseNew(Dense):
    def call(self, x, mask=None):
        # 1e-8 is used to prevent division by 0
        W_norm = self.W / (tf.sqrt(tf.reduce_sum(tf.square(self.W), axis=0, keep_dims=True)) + 1e-8)

        output = K.dot(x, W_norm)
        if self.bias:
            output += self.b
        return self.activation(output)
