from __future__ import absolute_import
from __future__ import print_function

from keras import backend as K
from keras.callbacks import Callback
from pkg_resources import parse_version


class TensorBoard(Callback):
    """ TensorBoard doesnt work with latest tensorflow from sources.
    This fixes some of the version checks.
    """

    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False):
        super(TensorBoard, self).__init__()
        if K._BACKEND != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images

    def _set_model(self, model):
        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        self.model = model
        self.sess = KTF.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.histogram_summary(weight.name, weight)

                    if self.write_images:
                        w_img = tf.squeeze(weight)

                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)

                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)

                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)

                        tf.image_summary(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.histogram_summary('{}_out'.format(layer.name),
                                         layer.output)
        if parse_version(tf.__version__) >= parse_version('0.12.0'):
            self.merged = tf.summary.merge_all()
        else:
            self.merged = tf.summary.merge_all()
        if self.write_graph:
            if parse_version(tf.__version__) >= parse_version('0.12.0'):
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            elif parse_version(tf.__version__) >= parse_version('0.8.0'):
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph)
            else:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
        else:
            if parse_version(tf.__version__) >= parse_version('0.12.0'):
                self.writer = tf.summary.FileWriter(self.log_dir)
            else:
                self.writer = tf.train.SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        import tensorflow as tf

        if self.model.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.model.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.model.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()


import cv2
from skimage import io


def load_img(path, grayscale=False, target_size=None):
    """Utility function to load an image from disk.

    Args:
      path: The image file path.
      grayscale: True to convert to grayscale image (Default value = False)
      target_size: (w, h) to resize. (Default value = None)

    Returns:
        The loaded numpy image.
    """
    img = io.imread(path, grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img
