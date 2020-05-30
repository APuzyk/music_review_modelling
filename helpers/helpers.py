import numpy as np


def sort_l_x_by_l_y(x, y, descending=False):
    return [x for _, x in sorted(zip(y, x), key=lambda pair: pair[0],reverse=descending)]


def one_hot(x):
    o = np.zeros((x.size, x.max() + 1))
    o[np.arange(x.size), x] = 1
    return o
