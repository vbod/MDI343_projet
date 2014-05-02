from __future__ import print_function

import numpy as np
import utils
import matplotlib.pyplot as plt
import random as rd

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
rbm_layer_1.n_iter = 25
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
rbm_layer_2.n_iter = 20
rbm_layer_2.n_components = 100
# Training RBM
rbm_layer_2.fit(H1_train)