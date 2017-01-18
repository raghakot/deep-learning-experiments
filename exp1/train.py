from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.layers import Convolution2D as conv_old
from layers import Convolution2DNew as conv_new
from cifar10 import train


def get_model(conv=conv_old):
    model = Sequential()
    model.add(conv(32, 3, 3, border_mode='same', activation='relu',
                   bias=False, input_shape=(32, 32, 3)))
    model.add(conv(32, 3, 3, bias=False, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(conv(64, 3, 3, bias=False, activation='relu', border_mode='same'))
    model.add(conv(64, 3, 3, bias=False, activation='relu'))

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

    names = ['baseline', 'new']
    convs = [conv_old, conv_new]

    for i in range(len(names)):
        model = get_model(convs[i])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train(names[i], model, nb_epoch=100)
