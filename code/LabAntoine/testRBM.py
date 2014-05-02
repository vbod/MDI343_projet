from __future__ import print_function

import numpy as np
import utils
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics, grid_search
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from pprint import pprint
from sklearn.utils import check_random_state

###############################################################################
# Settings

# Chargement des digits
X, Y = utils.load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression() # pour comparaison avec RBM + regression logistique
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
# best 0.04 25
parameters = {'rbm__learning_rate': np.linspace(0.04, 0.05, num=10)}
# rbm.learning_rate = 0.041
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
# penalite pour la regression logistic
logistic.C = 6000.0

gridSearch = grid_search.GridSearchCV(classifier, parameters)
# Training RBM-Logistic Pipeline
print("Performing grid search...")
print("pipeline:", [name for name, _ in classifier.steps])
print("parameters:")
pprint(parameters)
print(gridSearch.fit(X_train, Y_train))
print("Best score: %0.3f" % gridSearch.best_score_)
print("Best parameters set:")
best_parameters = gridSearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
# # Training Logistic regression
# logistic_classifier = linear_model.LogisticRegression(C=100.0)
# logistic_classifier.fit(X_train, Y_train)

# ###############################################################################
# # Evaluation

# print()
# print("Logistic regression using RBM features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         classifier.predict(X_test))))

# print("Logistic regression using raw pixel features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         logistic_classifier.predict(X_test))))

# ###############################################################################
# # Plotting

# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(rbm.components_):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle('100 components extracted by RBM1', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# plt.show()