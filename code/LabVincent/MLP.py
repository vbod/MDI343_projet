# -*- coding: utf-8 -*-
import numpy as np

tanh = lambda x: np.tanh(x)
tanh_derivative = lambda x: 1.-x**2
logistic = lambda x: 1/(1 + np.exp(-x))
logistic_derivative = lambda x: logistic(x)*(1-logistic(x))
    
class MLP:
    def __init__(self, layers, weights = None, activation = 'tanh'):
        """
        Constructeur du multi-layer perceptron (MLP). Si weights = None est 
        envoyé alors les poids ne sont pas donnés en entrée, on fait un MLP 
        classique - sans pre-training. Le cas échéant, les poids doivent être 
        tirés aléatoirement dans [-sqrt(6/(n_in + n_out)), 
        sqrt(6/(n_in + n_out))] pour une activation tanh (Bengio, Glorot,  
        Understanding the difficulty of training deep feedforward
        neuralnetworks) ou 4*[-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out))] 
        pour activation logistique. Ici n_in est le nombre de composant à la 
        couche i-1 et n_out de composant à la couche i. Dans le cas de DBN e.g.
        envoyer la matrice de poids directement dans weights.
        
        Input :
            - layers : liste contenant le nombre de composant dans chaque 
              couche. Doit contenir au moins deux couches (input - output).
            - activation : fonction d'activation, soit logistique soit tangente
              hyperbolique.
            - weights : matrice de poids, None si poids non déterminé au départ.
        
        Output : 
            - self : objet.
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative
        
        self.layers = layers
        
        if weights == None:
            self.weights = []
            # tire aléatoirement les poids d'entrée selon l'activation. Pour 
            # toutes les couches sauf la dernière, on rajoute un offset. 
            for i in range(1, len(layers)-1):
                low = -np.sqrt(6. / (layers[i-1] + layers[i]))
                high = np.sqrt(6. / (layers[i-1] + layers[i]))
                if activation == 'logistic':
                    low *= 4
                    high *= 4
                self.weights.append((high-low)*np.random.random(
                                    (layers[i-1]+1, layers[i]+1)) + low)
                                    
            self.weights.append((high-low)*np.random.random((layers[i]+1, 
                                layers[i+1])) + low)
        else:
            self.weights = weights
    
    
    def fit(self, X, y, epochs = 1000, learning_rate = .2):
        """
        Calcule les poids en fonction des données fournies (supervisé). La 
        méthode utilisée est une descente de gradient stochastique - choix 
        aléatoire d'un échantillon de l'ensemble d'apprentissage, calcul de 
        la sortie puis une étape de backpropagation associée à cet échantillon,
        i.e. calcul d'un delta qui comptabilise les erreurs commise. On répète
        un nombre epochs de fois.
        
        Input : 
            - X : ensemble d'entrainement, np.array (n_samples, n_features)
            - y : label d'entrainement, np.array (n_samples,)
            - epochs : nombre de fois à répeter la descente de gradient 
              stochastique
            - learning_rate : taux de mise à jour des poids
                        
        """
        # Rajoute le terme de biais à la couche d'entrée
        X = np.hstack((X, np.ones((int(X.shape[0]),1))))
        
        for k in range(epochs):
            i = np.random.randint(int(X.shape[0]))
            states = [X[i]] # ligne features du sample i
            
            # passage forward
            for j in range(len(self.weights)):
                states.append(self.activation(np.dot(states[j], 
                              self.weights[j])))
                       
            # passage backpropagation
            error = y[i] - states[-1]
            deltas = [error*self.activation_deriv(states[-1])]
            for j in range(len(states)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T)*
                              self.activation_deriv(states[j]))
            deltas.reverse()
            
            # update des poids
            for i in range(len(self.weights)):
                layer = np.atleast_2d(states[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate*layer.T.dot(delta)
    
    
    def predict(self, X):
        """
        Effectue de la prédiction sur le MLP appris par la méthode fit et un 
        ensemble de test dans X. 
        
        Input :
            - X : matrice de test, np.array (n_test, n_features)
            
        Ouput :
            - res : matrice contenant les résultats, np.array 
              (n_test, n_layer_end) où n_layer end est le nombre à la dernière
              couche
        """
        X = np.hstack((X, np.ones((int(X.shape[0]),1))))
        res = np.zeros((int(X.shape[0]), int(self.layers[-1])))
        for i in range(int(X.shape[0])):
            a = X[i]
            for j in range(0, len(self.weights)):
                a = self.activation(np.dot(a, self.weights[j]))
            res[i] = a
        return res


if __name__ == '__main__':    
    mlp = MLP([2,3,2])
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    mlp.fit(X, y)
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1,1]])
    res = mlp.predict(X_test)
    print res
    