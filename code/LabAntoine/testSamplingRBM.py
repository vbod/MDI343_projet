from __future__ import print_function

import numpy as np
import utils
import matplotlib.pyplot as plt
import random as rd
import MLP

from sklearn import linear_model, metrics, grid_search
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from pprint import pprint
from sklearn.utils import check_random_state

###############################################################################
# Settings
n_sample_second_layer_training = 10

# Chargement des digits
X, Y = utils.load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
rbm_layer_1 = BernoulliRBM(random_state=0, verbose=True)
rbm_layer_2 = BernoulliRBM(random_state=0, verbose=True)
logistic = linear_model.LogisticRegression() # pour comparaison avec RBM + regression logistique
###############################################################################
# Training du premier rbm
rbm_layer_1.learning_rate = 0.04
rbm_layer_1.n_iter = 2
rbm_layer_1.n_components = 100
# Training RBM
rbm_layer_1.fit(X_train)

# creation d'une base de train a partir d'echantillonnage
# de variable cachees du premier rbm
n_sample_second_layer_training = 2*int(X.shape[0])
H1_train = np.zeros(shape=(n_sample_second_layer_training, rbm_layer_1.n_components))
comp = 0
while (comp < n_sample_second_layer_training):
	rng = check_random_state(rbm_layer_1.random_state)
	randTemp = rd.randint(0, X.shape[0] - 1)
	H1_train[comp] = rbm_layer_1._sample_hiddens(X[randTemp], rng)
	comp = comp + 1

# Training du second rbm
rbm_layer_2.learning_rate = 0.06
rbm_layer_2.n_iter = 2
rbm_layer_2.n_components = 100
# Training RBM
rbm_layer_2.fit(H1_train)

rbm1w = rbm_layer_1.components_.T
bias1h = rbm_layer_1.intercept_hidden_
bias1h = bias1h.reshape(bias1h.size, 1)
bias1v = rbm_layer_1.intercept_visible_
bias1v = bias1v.reshape(bias1v.size, 1)

rbm2w = rbm_layer_2.components_.T
bias2h = rbm_layer_2.intercept_hidden_
bias2h = bias2h.reshape(bias2h.size, 1)
bias2v = rbm_layer_2.intercept_visible_
bias2v = bias2v.reshape(bias2v.size, 1)

W1 = np.vstack((np.hstack((rbm1w, bias1v)), np.hstack((bias1h.T, np.zeros(shape=(1, 1))))))
W2 = np.vstack((np.hstack((rbm2w, bias2v)), np.hstack((bias2h.T, np.zeros(shape=(1, 1))))))

weights = [W1, W2]
layers = [64+1, rbm_layer_1.n_components+1, rbm_layer_2.n_components+1]

print(type(weights))

print(W1)
print(W1.shape)
print(W2)
print(W2.shape)