import cv2

from collections import defaultdict
from skimage import io
from keras.callbacks import Callback


class ModelLogger(Callback):
    """Custom callback to log tensors within the model.
    """

    @staticmethod
    def get_logger(layer_type, log_lambda):
        """Builds the logger object.

        :param layer_type: The keras layer type to log.
        :param log_lambda: A function with inputs (layer, idx) for logging.
            layer is an instance of `layer_type`.
            idx indicates the count of the layer. idx = 1 would mean that `layer` is second from top.
        :return: Logger instance.
        """
        return {
            'type': layer_type,
            'fn': log_lambda
        }

    def __init__(self, loggers):
        """`Generate individual logger using `get_logger`
        """
        self.loggers = loggers
        super(ModelLogger, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        type_idx_map = defaultdict(int)

        for layer in self.model.layers:
            for logger in self.loggers:
                if isinstance(layer, logger['type']):
                    logger['fn'](layer, type_idx_map[type(layer)])
            type_idx_map[type(layer)] += 1


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
