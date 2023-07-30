import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, eta=0.01, epochs=50, is_verbose = False):
        ''' Initialize the Adaline model '''
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    def predict(self, x):
        ''' Make predictions using the current weights '''
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)

        return np.where(self.get_activation(x_1) > 0, 1, -1)
        
    def get_activation(self, x):
        ''' Calculate the activation for the given input and current weights '''
        activation = np.dot(x, self.w)
        return activation
     
    def fit(self, X, y):
        ''' Train the Adaline model using the input data and target labels '''
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
 
        self.w = np.random.rand(X_1.shape[1])
        
        for epoch in range(self.epochs):
            MSE_error = 0
            
            activation = self.get_activation(X_1)
            gradient = self.eta * np.dot((y - activation), X_1)
            self.w += gradient
                
            MSE_error = np.mean((y - activation)**2)
                
            self.list_of_errors.append(MSE_error)
            