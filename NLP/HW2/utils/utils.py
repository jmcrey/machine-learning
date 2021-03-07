#!/usr/bin/env python

import numpy as np


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x.astype(float) / norm

    ### END YOUR CODE
    return x


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    ### YOUR CODE HERE

    exps = np.exp(x - np.max(x))
    x = exps / np.sum(exps, axis=1, keepdims=True)

    ### END YOUR CODE
    return x
