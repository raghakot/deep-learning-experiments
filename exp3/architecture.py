import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Lambda, Layer


class Connection(Layer):
    """Takes a list of inputs, resizes them to the same shape, and outputs a weighted merge.
    """
    def __init__(self, init_value=0.5, merge_mode='concat',
                 resize_interpolation=tf.image.ResizeMethod.BILINEAR,
                 shared_weights=True):
        """Creates a connection that encapsulates weighted connection of input feature maps.

        :param init_value: The init value for connection weight
        :param merge_mode: Defines how feature maps from inputs are aggregated.
        :param resize_interpolation: The downscaling interpolation to use. Instance of `tf.image.ResizeMethod`.
            Note that ResizeMethod.AREA will fail as its gradient is not yet implemented.
        :param shared_weights: True to share the same weight for all feature maps from inputs[i].
        False creates a separate weight per feature map.
        """
        self.init_value = init_value
        self.merge_mode = merge_mode
        self.resize_interpolation = resize_interpolation
        self.shared_weights = shared_weights
        super(Connection, self).__init__()

    def _ensure_same_size(self, inputs):
        """Ensures that all inputs match last input size.
        """
        # Find min (row, col) value and resize all inputs to that value.
        rows = min([K.int_shape(x)[1] for x in inputs])
        cols = min([K.int_shape(x)[2] for x in inputs])
        return [tf.image.resize_images(x, [rows, cols], self.resize_interpolation) for x in inputs]

    def _merge(self, inputs):
        """Define other merge ops like [+, X, Avg] here.
        """
        if self.merge_mode == 'concat':
            # inputs are already stacked.
            return inputs
        else:
            raise RuntimeError('mode {} is invalid'.format(self.merge_mode))

    def build(self, input_shape):
        """ Create trainable weights for this connection
        """
        # Number of trainable weights = sum of all filters in `input_shape`
        nb_filters = sum([s[3] for s in input_shape])

        # Create shared weights for all filters within an input layer.
        if self.shared_weights:
            weights = []
            for shape in input_shape:
                # Create shared weight, make nb_filter number of clones.
                shared_w = K.variable(np.ones(1) * self.init_value)
                for _ in range(shape[3]):
                    weights.append(shared_w)
            self.W = K.concatenate(weights)
        else:
            self.W = K.variable(np.ones(shape=nb_filters) * self.init_value)

        self._trainable_weights.append(self.W)
        super(Connection, self).build(input_shape)

    def call(self, layer_inputs, mask=None):
        # Resize all inputs to same size.
        resized_inputs = self._ensure_same_size(layer_inputs)

        # Compute sigmoid weighted inputs
        stacked = K.concatenate(resized_inputs, axis=-1)
        weighted = stacked * K.sigmoid(self.W)

        # Merge according to provided merge strategy.
        merged = self._merge(weighted)

        # Cache this for use in `get_output_shape_for`
        self._out_shape = K.int_shape(merged)
        return merged

    def get_output_shape_for(self, input_shape):
        return self._out_shape


def get_fully_connected(x, layers, **connection_args):
    """Creates a fully connected net consisting of layers that generate 4D output.
    """
    inputs = [x]
    for layer in layers:
        x = layer(x)
        inputs.append(x)
        x = Connection(**connection_args)(inputs)
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
