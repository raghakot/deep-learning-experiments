from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.layers import Convolution2D as conv_old
from keras.callbacks import EarlyStopping

from layers import Convolution2D_4 as conv_4
from layers import Convolution2D_8 as conv_8
from cifar10 import train


def get_model(convs=None):
    if convs is None:
        convs = [conv_old, conv_old, conv_old, conv_old]

    model = Sequential()
    model.add(convs[0](32, 3, 3, border_mode='same', activation='relu',
                       bias=False, input_shape=(32, 32, 3)))
    model.add(convs[1](32, 3, 3, bias=False, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(convs[2](64, 3, 3, bias=False, border_mode='same', activation='relu'))
    model.add(convs[3](64, 3, 3, bias=False, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


if __name__ == '__main__':
    # Cleanup old weight dir.
    import os, shutil
    shutil.rmtree('weights', ignore_errors=True)
    os.makedirs('weights')

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    names = ['baseline', '8_rot_4', '8_rot_3', '8_rot_2', '8_rot_1', '4_rot_4', '4_rot_3', '4_rot_2', '4_rot_1']
    convs = [
        None,
        [conv_8, conv_8, conv_8, conv_8],
        [conv_old, conv_8, conv_8, conv_8],
        [conv_old, conv_old, conv_8, conv_8],
        [conv_old, conv_old, conv_old, conv_8],
        [conv_4, conv_4, conv_4, conv_4],
        [conv_old, conv_4, conv_4, conv_4],
        [conv_old, conv_old, conv_4, conv_4],
        [conv_old, conv_old, conv_old, conv_4],
    ]

    for i in range(len(names)):
        model = get_model(convs[i])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train(names[i], model, callbacks, nb_epoch=200)
        print('-' * 20)
