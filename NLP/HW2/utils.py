"""
Date: March 6, 2021
Purpose: Implements the numerically stable version of the softmax function
"""

__version__ = '0.0.1'
__author__ = 'Jeremiah McReynolds'

import numpy as np


def softmax(xi: np.ndarray):
    """ Computes the numerically stable version of the softmax function """
    exps = np.exp(xi - np.max(xi))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss():
    """ Implements the cross-entropy loss function """
    ...


def cross_entropy_cost():
    """ Implements the cross-entropy cost function """