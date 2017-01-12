""" Random stuff used to generate figures as needed.
"""

from __future__ import print_function
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def outer_slice(x):
    return np.r_[x[0], x[1:-1, -1], x[-1, :0:-1], x[-1:0:-1, 0]]


def rotate_steps(x, shift):
    out = np.empty_like(x)
    N = x.shape[0]
    idx = np.arange(x.size).reshape(x.shape)
    for n in range((N + 1) // 2):
        sliced_idx = outer_slice(idx[n:N - n, n:N - n])
        out.ravel()[sliced_idx] = np.roll(np.take(x, sliced_idx), shift)
    return out


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


if __name__ == '__main__':
    w = np.array([
        [0.2, 1.0, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 1.0, 1.0]
    ])

    w_list = [w]
    for i in range(7):
        w = rotate_steps(w, 1)
        w_list.append(w)

    nice_imshow(pl.gca(), make_mosaic(np.array(w_list), 2, 4), vmin=0, vmax=1, cmap=cm.binary)
    pl.show()
