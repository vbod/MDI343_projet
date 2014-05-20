from __future__ import print_function

import numpy as np
import utils
import matplotlib.pyplot as plt
import random as rd
import MLP
import shelve
import time

from sklearn.datasets import fetch_mldata
from sklearn import svm, datasets
from sklearn import linear_model, metrics, grid_search, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from pprint import pprint
from sklearn.utils import check_random_state

###############################################################################
# Settings
n_sample_second_layer_training = 10
X, Y = utils.load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# ############################
# Apprentissage du premier rbm1
# ############################
rbm_layer_1 = BernoulliRBM(random_state=0, verbose=True)
rbm_layer_1.learning_rate = 0.06
rbm_layer_1.n_iter = 25
rbm_layer_1.n_components = 100
rbm_layer_1.fit(X_train)

# ############################
# Sampling pour l'entree du deuxieme rbm
# ############################
n_sample_second_layer_training = int(X.shape[0])
print(n_sample_second_layer_training)
H1_train = np.zeros(shape=(n_sample_second_layer_training, rbm_layer_1.n_components))
H1_label_train = np.zeros(shape=(n_sample_second_layer_training, 1))
comp = 0
while (comp < n_sample_second_layer_training):
	rng = check_random_state(rbm_layer_1.random_state)
	randTemp = rd.randint(0, X.shape[0] - 1)
	H1_train[comp] = rbm_layer_1._sample_hiddens(X[randTemp], rng)
	H1_label_train[comp] = Y[randTemp]
	comp = comp + 1
H1_label_train = H1_label_train.astype('int')
# ############################
# Preparation du grid search
# ############################
rbm_layer_2 = BernoulliRBM(random_state=0, verbose=True)
rbm_layer_2.n_components = 100
logistic = linear_model.LogisticRegression()
classifier = Pipeline(steps=[('rbm_layer_2', rbm_layer_2), ('logistic', logistic)])
parameters = {'rbm_layer_2__learning_rate': np.linspace(0.01, 0.04, num=10),
			'rbm_layer_2__n_iter': np.linspace(10, 50, num=3).astype('int')}
gridSearch = grid_search.GridSearchCV(classifier, parameters)
print(gridSearch.fit(H1_train, H1_label_train))
print("Best score: %0.3f" % gridSearch.best_score_)
print("Best parameters set:")
best_parameters = gridSearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))

# # Training RBM
# print("Debut training RBM1")
# print(H1_train.shape)
# t0 = time.clock()
# rbm_layer_2.fit(H1_train)
# print(time.clock() - t0)
# rbm1w = rbm_layer_1.components_.T
# bias1h = rbm_layer_1.intercept_hidden_
# bias1h = bias1h.reshape(bias1h.size, 1)
# bias1v = rbm_layer_1.intercept_visible_
# bias1v = bias1v.reshape(bias1v.size, 1)

# rbm2w = rbm_layer_2.components_.T
# bias2h = rbm_layer_2.intercept_hidden_
# bias2h = bias2h.reshape(bias2h.size, 1)
# bias2v = rbm_layer_2.intercept_visible_
# bias2v = bias2v.reshape(bias2v.size, 1)

# W1 = np.vstack((np.hstack((rbm1w, bias1v)), np.hstack((bias1h.T, np.zeros(shape=(1, 1))))))
# W2 = np.vstack((np.hstack((rbm2w, bias2v)), np.hstack((bias2h.T, np.zeros(shape=(1, 1))))))
# weights = [W1, W2]

# layers = [64+1, rbm_layer_1.n_components+1, rbm_layer_2.n_components+1]

# # layers = [64, 100, 100]
# print(X_train.shape)
# print(X_test.shape)
# print("Training MLP")
# t0 = time.clock()
# mlp = MLP.MLP(layers, weights=weights)
# mlp.fit(X_train, Y_train, epochs=1000)
# print(time.clock() - t0)


# print("Calcul nouvelles representations")
# print("Train")
# t0 = time.clock()
# X_train_new = np.zeros((X_train.shape[0], sum(layers)))
# for i in range(int(X_train.shape[0])):
# 	if (i % 1000 == 0):
# 		print(i)
# 	a = np.hstack((X_train[i].reshape((1, X_train.shape[1])), np.ones((1, 1))))
# 	resTemp = a
# 	for j in range(0, len(mlp.weights)):
# 		resTemp = mlp.activation(np.dot(resTemp, mlp.weights[j]))
# 		a = np.hstack((a, resTemp))
# 	X_train_new[i, :] = a
# print(time.clock() - t0)
# print("Test")
# t0 = time.clock()
# X_test_new = np.zeros((X_test.shape[0], sum(layers)))
# for i in range(int(X_test.shape[0])):
# 	if (i % 1000 == 0):
# 		print(i)
# 	a = np.hstack((X_test[i].reshape((1, X_test.shape[1])), np.ones((1, 1))))
# 	resTemp = a
# 	for j in range(0, len(mlp.weights)):
# 		resTemp = mlp.activation(np.dot(resTemp, mlp.weights[j]))
# 		a = np.hstack((a, resTemp))
# 	X_test_new[i] = a
# print(time.clock() - t0)

# print("Training SVM")
# t0 = time.clock()
# clf = svm.SVC()
# clf.fit(X_train_new, Y_train)  
# print(time.clock() - t0)

# print("SVM using DBN features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         clf.predict(X_test_new))))

# print("Calcul regression logistique")
# t0 = time.clock()
# logistic = linear_model.LogisticRegression()
# logistic.C = 6000
# X_train_new = preprocessing.scale(X_train_new)
# X_test_new = preprocessing.scale(X_test_new)

# logistic.fit(X_train_new, Y_train)
# print(time.clock() - t0)
# # print("Score")
# # score = logistic.score(X_test_new, Y_test)

# print("Logistic regression using DBN features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         logistic.predict(X_test_new))))

# # variablesTemp = shelve.open('sauvegardeVariablesTemp')
# # variablesTemp['RBM1'] = rbm_layer_1
# # variablesTemp['RBM2'] = rbm_layer_2
# # variablesTemp['MLP'] = mlp.weights
# # 	print("resTemlogistic.C = 6000.0p")
# # 	print(resTemp)
# # 	print([item for sublist in resTemp for item in sublist])
# # 	res[i, :] = np.aray([item for sublist in resTemp for item in sublist])

# # print(res.shape)