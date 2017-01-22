from __future__ import print_function

from keras.models import Model
from keras.layers import Dropout, MaxPooling2D, Dense, Input
from keras.layers import Convolution2D

from cifar10 import train
from architecture import get_fully_connected, get_sequential, flatten


def get_model(connection_weight_init=1., connection_merge_mode='concat'):
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
    if connection_weight_init is None:
        out = get_sequential(x, layers)
    else:
        out = get_fully_connected(x, layers, connection_weight_init, connection_merge_mode)

    # Complete the rest of the network with Dense layers.
    out = flatten(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(10, activation='softmax')(out)
    return Model(input=x, output=out)


def visualize_model():
    model = get_model()

    from keras.utils.visualize_util import model_to_dot
    dot = model_to_dot(model)
    dot.set('concentrate', False)
    dot.write('model.png', format='png')


if __name__ == '__main__':
    # Cleanup old weight dir.
    import os, shutil
    shutil.rmtree('weights', ignore_errors=True)
    os.makedirs('weights')

    names = ['baseline', 'fc_init_1_merge_concat', 'fc_init_0_5_merge_concat', 'fc_init_0_merge_concat']
    params = [(None, None), (1., 'concat'), (0.5, 'concat'), (0., 'concat')]

    for i in range(len(names)):
        model = get_model(*params[i])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train(names[i], model, nb_epoch=75)
