
import numpy as np


def knn(vec, mat, k):
    """

    :param vec: Word Vector to compare
    :param mat: All the possible neighbors
    :param k: How many neighbors to return
    :return: List of Indicies, not sorted
    """
    n, d = mat.shape
    assert d == vec.shape[0]

    x = mat.dot(vec)
    y = np.sqrt(np.sum(mat ** 2, axis=1)) * np.sqrt(np.sum(vec ** 2))
    cosine = x / y
    neighbors = list(np.argsort(cosine)[-k:])
    neighbors.reverse()
    return neighbors


def print_neighbors(word, neighbors, tokens):
    """ Prints all the neighbors """
    nearest = [key for i in neighbors for key, value in tokens.items() if value == i]
    print(f"Nearest neighbors for '{word}': {', '.join(nearest)} ")


def knn_wrapper(words, tokens, word_vectors, k):
    """ Wrapper for the KNN function """
    for word in words:
        idx = tokens[word]
        vec = word_vectors[idx]
        neighbors = knn(vec, word_vectors, k)
        print_neighbors(word, neighbors, tokens)

