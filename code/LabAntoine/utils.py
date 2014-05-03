from scipy.ndimage import convolve
from sklearn import datasets
import numpy as np
from sklearn.datasets import fetch_mldata

##############################
#            DATA            #
##############################

def load_data():
    """
    Charge les digits avec convolution et normalisation 0-1

    Output :
        - X digits
        - Y labels
    """
    print("hopla")
    # mnist = fetch_mldata('MNIST original', data_home='../data')
    # print(mnist.data.shape)
    digits = datasets.load_digits()
    # X = mnist.data
    # Y = mnist.Y
    X = np.asarray(digits.data, 'float32')
    X, Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)
    return X, Y

def nudge_dataset(X, Y):
    """
    Multiplie la taille de X par 5 en translatant les images

    Input :
        - X digits
        - Y labels

    Input :
        - X digits
        - Y labels
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y