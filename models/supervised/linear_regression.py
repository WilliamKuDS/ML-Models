import numpy as np
import random

#Initial Attempt with no inital baseline
class LinearRegression:
    def __init__(self):
        self.theta = None
        self.steps = 500000

    def fit(self, X, y, *args):
        # Will fix for optional args
        learning_rate = args[0]
        # Get the number of features
        n = X.shape[1]
        # Initialize theta as zero array with the size of features
        self.theta = np.zeros(n)
        # Iterate till max steps
        for _ in range(self.steps):
            #Gradient Descent update, j = number of features
            for j in range(n):
                j_cost = self.gd_cost(X, y, X[:,j])
                self.theta[j] = self.theta[j] - learning_rate*j_cost
        # Need to implement convergence statement here

    def predict(self, X):
        return np.dot(X, self.theta)

    #Cost Function for Batch GD (very slow)
    def b_cost(self, X, y, xj):
        min_cost = float('inf')
        cost = 0
        m = X.shape[0]
        for i in range(m):
            cost += np.dot(np.dot(self.theta, X[i]) - y[i], xj[i])
            min_cost = min(min_cost, cost)
        return min_cost

    #Cost Function for Stocastic GD
    def gd_cost(self, X, y, xj):
        # Pick a random training data point (i)
        i = random.randrange(0, X.shape[0])
        # Returns cost function of ((xi * theta) - yi) x[j][i])
        return ((np.dot(X[i], self.theta) - y[i]) * xj[i])

    #Cost Function for Mini-Batch GD


