from scipy.ndimage import convolve
from sklearn import datasets
import numpy as np

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
    digits = datasets.load_digits()
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

# ##############################
# #            RBMs            #
# ##############################

# def gibbs(rbm, v):
#         """
#         sample par 1-Divergence à partir de v

#         Input :
#             - rbm : le classifieur
#             - v : la donnee a partir de laquel on part pour la 1-D

#         Output :
#             - v : sampled
#         """
#         rng = check_random_state(rbm.random_state)
#         h_ = rbm._sample_hiddens(v, rng)
#         v_ = rbm._sample_visibles(h_, rng)

#         return v_

# def _sample_visibles(rbm, h, rng):
#         """
#         sample grace à P(v|h).

#         Input :
#             - rbm : le classifieur

#         rng : RandomState


#         Returns
#         -------
#         v : array-like, shape (n_samples, n_features)
#             Values of the visible layer.
#         """
#         p = logistic_sigmoid(np.dot(h, self.components_)
#                              + self.intercept_visible_)
#         p[rng.uniform(size=p.shape) < p] = 1.
#         return np.floor(p, p)