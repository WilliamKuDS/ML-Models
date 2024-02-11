import numpy as np
import random

#Initial Attempt with no inital baseline

class LinearRegression:
    def __init__(self):
        self.theta = None
        self.steps = 10000

    def fit(self, X, y, *args):
        #args = [learning rate, steps, ]
        learning_rate = args[0]
        #if args[1] is None:
            #self.steps = args[1]
        self.theta = np.zeros(X.shape)
        n = X.shape[1]
        cost = float('inf')
        for a in range(self.steps):
            #Gradient Descent update
            for j in range(n):
                j_cost = self.gd_cost(X, y, X[j])
                if j_cost < cost:
                    cost = j_cost
                    self.theta[j] = self.theta[j] - learning_rate * cost
                else:
                    break


    def predict(self, X):
        y = self.theta * X
        return y

    #Cost Function for Batch GD (very slow)
    def b_cost(self, X, y, xj):
        min_cost = float('inf')
        cost = 0
        m = X.shape[0]
        for i in range(m):
            cost += (self.theta * X[i] - y[i]) * xj[i]
            min_cost = min(min_cost, cost)

        return min_cost

    #Cost Function for Stocastic GD
    def gd_cost(self, X, y, xj):
        i = random.randrange(0, X.shape[0])
        return self.theta[i] * X[i] - y[i] * xj[i]

    #Cost Function for Mini-Batch GD

