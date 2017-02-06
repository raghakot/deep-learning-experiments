from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

from keras.layers import Convolution2D as conv_old
from layers import Convolution2DNew as conv_new
from cifar10 import train


def get_model(conv=conv_old, l2_reg=1e-3):
    """ Builds an all CNN model with provided conv implementation.
    """
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(32, 32, 3)))
    model.add(conv(96, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(conv(96, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(conv(96, 3, 3, subsample=(2, 2), border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(conv(192, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(conv(192, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(conv(192, 3, 3, subsample=(2, 2), border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(conv(192, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(conv(192, 1, 1, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))
    model.add(conv(10, 1, 1, border_mode='same', W_regularizer=l2(l2_reg),
                   init='he_normal', activation='relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    # Cleanup old weight dir.
    import os, shutil
    shutil.rmtree('weights', ignore_errors=True)
    os.makedirs('weights')

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    names = ['baseline', 'norm_conv']
    convs = [conv_old, conv_new]

    for i in range(len(names)):
        model = get_model(convs[i])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        train(names[i], model, callbacks, nb_epoch=200)
        print('-' * 20)
