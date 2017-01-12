from keras.layers.convolutional import Convolution2D
from keras import backend as K
import tensorflow as tf


permutation = [[1, 0], [0, 0], [0, 1], [2, 0], [1, 1], [0, 2], [2, 1], [2, 2], [1, 2]]


def shift_rotate(w, shift=1):
    shape = w.get_shape()
    for i in range(shift):
        w = tf.reshape(tf.gather_nd(w, permutation), shape)
    return w


class Convolution2D_4(Convolution2D):
    def call(self, x, mask=None):
        w = self.W
        w_rot = [w]
        for i in range(3):
            w = shift_rotate(w, shift=2)
            w_rot.append(w)

        outputs = tf.stack([K.conv2d(x, w_i, strides=self.subsample,
                                     border_mode=self.border_mode,
                                     dim_ordering=self.dim_ordering,
                                     filter_shape=self.W_shape) for w_i in w_rot])

        output = K.max(outputs, 0)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output


class Convolution2D_8(Convolution2D):
    def call(self, x, mask=None):
        w = self.W
        w_rot = [w]
        for i in range(7):
            w = shift_rotate(w)
            w_rot.append(w)

        outputs = tf.stack([K.conv2d(x, w_i, strides=self.subsample,
                                     border_mode=self.border_mode,
                                     dim_ordering=self.dim_ordering,
                                     filter_shape=self.W_shape) for w_i in w_rot])

        output = K.max(outputs, 0)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output
