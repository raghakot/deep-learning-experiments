import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage
from train import get_model
from cifar10 import X_test, y_test

from keras.layers import Convolution2D as conv_old
from layers import Convolution2D_4 as conv_4
from layers import Convolution2D_8 as conv_8


def get_probs_matched(imgs, model, idx):
    probs = model.predict_proba(imgs, verbose=0)
    preds = probs.argmax(axis=-1)
    probs = probs[:, y_test[idx][0]]
    matched = (len(np.where(preds == y_test[idx][0])[0]) * 100.0) / len(imgs)
    return probs, matched


def compare(test_id, angles, models):
    img = X_test[test_id]
    imgs = np.array([ndimage.rotate(img, rot, reshape=False) for rot in angles])

    all_probs = []
    all_matched = []
    for model in models:
        probs, matched = get_probs_matched(imgs, model, test_id)
        all_probs.append(probs)
        all_matched.append(matched)

    return all_probs, all_matched


def plot_multi(names, models, angles, runs=1000):
    indices = np.random.permutation(len(X_test))[:runs]

    matched_all = []
    for i, idx in enumerate(indices):
        print("Processing {}/{}".format(i, len(indices)))
        probs, matched = compare(idx, angles, models)
        matched_all.append(matched)

    matched_all = np.array(matched_all)
    df = pd.DataFrame.from_items([(names[i], matched_all[:, i]) for i in range(len(names))])
    sb.boxplot(data=df)
    plt.show()


def plot_single(names, models, angles, test_id):
    all_probs, all_matched = compare(test_id, angles, models)
    legends = []
    for i, probs in enumerate(all_probs):
        plt.plot(angles, probs)
        legends.append('{} {:.2f}%'.format(names[i], all_matched[i]))

    plt.ylabel('Prediction probability of correct class')
    plt.legend(legends, loc=9, bbox_to_anchor=(0.5, -0.05), ncol=len(names))
    plt.show()


if __name__ == '__main__':
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

    models = []
    for i in range(len(names)):
        model = get_model(convs[i])
        model.load_weights('./weights/{}.hdf5'.format(names[i]))
        models.append(model)

    angles = np.arange(0, 360, 1)
    plot_single(names, models, angles=angles, test_id=5)
    # plot_multi(names, models, angles=angles, runs=1000)
