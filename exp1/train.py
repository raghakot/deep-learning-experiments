from __future__ import print_function


from keras.models import Model
from keras.layers import Dropout, GlobalAveragePooling2D, Activation, Input, MaxPooling2D
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.layers import Convolution2D as conv_old
from layers import Convolution2DNew as conv_new
from cifar10 import train


def get_model(conv=conv_old, l2_reg=1e-3):
    """ Builds an all CNN model with provided conv implementation.
    Using MaxPooling instead of striding conv to save on params.
    """
    inp = Input(shape=(32, 32, 3))
    x = inp

    x = Dropout(0.2)(x)
    x = conv(96, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)
    x = conv(96, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)

    x = conv(192, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)
    x = conv(192, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)

    x = conv(192, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)
    x = conv(192, 1, 1, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)
    x = conv(10, 1, 1, border_mode='same', W_regularizer=l2(l2_reg),
             activation='relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return Model(input=inp, output=x)


if __name__ == '__main__':
    # Cleanup old weight dir.
    import os, shutil
    shutil.rmtree('weights', ignore_errors=True)
    os.makedirs('weights')

    names = ['baseline', 'norm_conv']
    convs = [conv_old, conv_new]
    opt = Adam(lr=1e-4)

    for i in range(len(names)):
        model = get_model(convs[i])
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        train(names[i], model, nb_epoch=250)
        print('-' * 20)
