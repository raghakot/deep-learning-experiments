import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import merge, Lambda, Layer


class Connection(Layer):
    """Takes a list of inputs, resizes them to the same shape, and outputs a weighted merge.
    """
    def __init__(self, nb_layer_inputs, init_value=0.5, merge_mode='concat', **kwargs):
        self.init_value = init_value
        self.merge_mode = merge_mode
        self.nb_layer_inputs = nb_layer_inputs
        super(Connection, self).__init__(**kwargs)

    def _ensure_same_size(self, inputs):
        """Ensures that all inputs match last input size.
        """
        # Find min (row, col) value and resize all inputs to that value.
        rows = min([K.int_shape(x)[1] for x in inputs])
        cols = min([K.int_shape(x)[2] for x in inputs])
        return [tf.image.resize_bilinear(x, [rows, cols]) for x in inputs]

    def build(self, input_shape):
        # Create a trainable weight variable for this connection
        self.W = [K.variable(np.ones(shape=1) * self.init_value) for _ in range(self.nb_layer_inputs)]
        self._trainable_weights.extend(self.W)
        super(Connection, self).build(input_shape)

    def call(self, layer_inputs, mask=None):
        # Resize all inputs to same size.
        resized_inputs = self._ensure_same_size(layer_inputs)

        # Compute sigmoid weighted inputs
        weighted_inputs = [resized_inputs[i] * K.sigmoid(self.W[i]) for i in range(self.nb_layer_inputs)]

        # Merge according to provided merge strategy.
        merged = merge(weighted_inputs, mode=self.merge_mode, concat_axis=-1)

        # Cache this for use in `get_output_shape_for`
        self._out_shape = K.int_shape(merged)
        return merged

    def get_output_shape_for(self, input_shape):
        return self._out_shape


def get_fully_connected(x, layers, connection_weight_init=1., connection_merge_mode='concat'):
    """Creates a fully connected net consisting of layers that generate 4D output.
    """
    inputs = [x]
    for layer in layers:
        x = layer(x)
        inputs.append(x)
        x = Connection(len(inputs), init_value=connection_weight_init, merge_mode=connection_merge_mode)(inputs)
    return x


def get_sequential(x, layers):
    """Creates a sequential chain from given layers.
    """
    for layer in layers:
        x = layer(x)
    return x


def flatten(x):
    """Avoids the Flatten bug in keras when used with tf backend as of 1/21/17
    """
    return Lambda(lambda x: tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())]))(x)
