from __future__ import print_function

import numpy as np

from keras.models import Model
from keras.layers import Dropout, MaxPooling2D, Dense, Input
from keras.layers import Convolution2D

from cifar10 import train
from architecture import get_fully_connected, get_sequential, flatten, Connection
from scipy.special import expit as sigmoid
from utils import ModelLogger
from collections import defaultdict


def get_model(**connection_params):
    layers = [
        Convolution2D(32, 3, 3, border_mode='same', activation='relu', bias=False),
        Convolution2D(32, 3, 3, bias=False, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Convolution2D(64, 3, 3, bias=False, activation='relu', border_mode='same'),
        Convolution2D(64, 3, 3, bias=False, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25)
    ]

    x = Input((32, 32, 3))

    # Interpret this as requesting for a regular sequential model
    if len(connection_params) == 0:
        out = get_sequential(x, layers)
    else:
        out = get_fully_connected(x, layers, **connection_params)

    # Complete the rest of the network with Dense layers.
    out = flatten(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(10, activation='softmax')(out)
    return Model(input=x, output=out)


def visualize_model(**connection_params):
    model = get_model(**connection_params)

    from keras.utils.visualize_util import model_to_dot
    dot = model_to_dot(model, show_shapes=True)
    dot.set('concentrate', False)
    dot.write('model.png', format='png')


def get_logging_callback(shared_weights=False):
    """Creates callback to log connection weights and returns the history.
    """
    history = defaultdict(list)

    def log_lambda(layer, idx):
        w = np.squeeze(np.array(layer.get_weights()))
        # Collapse all same feature weights into one value
        if shared_weights:
            w = np.unique(w)
        history[idx].append(sigmoid(w))

    connection_weight_logger = ModelLogger.get_logger(Connection, log_lambda)
    return ModelLogger([connection_weight_logger]), history


def plot_weight_history(dir, history):
    """Plots connection weight evolution from history
    """
    import seaborn as sb
    import numpy as np

    for idx in history.keys():
        fig = sb.plt.figure()
        ax = sb.plt.subplot(111)

        fig.suptitle("Connection_{} Weights".format(idx + 1), fontsize=16, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel("Sigmoid(W)", fontsize=14)

        w = np.array(history[idx])
        for i in range(w.shape[1]):
            ax.plot(w[:, i], label='w_{}'.format(i))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig('{}/connection_{}'.format(dir, idx))


if __name__ == '__main__':
    # Cleanup old weight dir.
    import os, shutil
    shutil.rmtree('weights', ignore_errors=True)
    os.makedirs('weights')
    shutil.rmtree('plots', ignore_errors=True)
    os.makedirs('plots')

    names = ['fc_nonshared_concat']
    params = [{'shared_weights': False}]

    for i in range(len(names)):
        # Connection weight history is not valid for baseline model.
        is_baseline = len(params[i]) == 0
        if is_baseline:
            callbacks = []
            history = None
        else:
            logging_callback, history = get_logging_callback()
            callbacks = [logging_callback]

        model = get_model(**params[i])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train(names[i], model, callbacks, nb_epoch=2)
        print('-' * 20)

        if not is_baseline:
            dir = 'plots/{}'.format(names[i])
            os.makedirs(dir)
            plot_weight_history(dir, history)
