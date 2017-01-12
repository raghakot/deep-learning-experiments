""" Sanity check shift_rotate implementation
"""

import numpy as np
import tensorflow as tf


w = np.array(
    [
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99]
            ],
            [
                [111, 222, 333],
                [444, 555, 666],
                [777, 888, 999]
            ]
        ],
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99]
            ],
            [
                [111, 222, 333],
                [444, 555, 666],
                [777, 888, 999]
            ]
        ]
    ]
)

w = np.moveaxis(w, (2, 3), (0, 1))

sess = tf.InteractiveSession()
# shape: (row, col, inp_nb, out_nb) -> (3, 3, 2, 3)
w = tf.constant(w)

from layers import shift_rotate
w1 = shift_rotate(w, 4)
print w1.eval()
