import numpy as np
import random
import time

#Initial Attempt with no inital baseline
class LinearRegression:
    def __init__(self):
        self.theta = None
        self.tolerance = 1e-6

    def fit(self, X, y, learning_rate=0.01, steps=1000000):
        start = time.time()
        # Get the number of features, where n = # of features
        n = X.shape[1]
        # Initialize theta as zero array with the size of features
        self.theta = np.zeros(n)
        prev_cost = float('inf')
        for step in range(steps):
            #Gradient Descent update, calculate cost, then update theta
            j_cost = self.gd_cost(X, y)
            self.theta = self.theta - learning_rate * j_cost
            # Convergence Test
            if step % 1000 == 0:
                current_cost = np.mean(self.theta)
                if np.abs(prev_cost - current_cost) < self.tolerance:
                    print('Model converged after {} steps'.format(step))
                    break
                prev_cost = current_cost
        print("Time: {}".format(time.time() - start))

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
    def gd_cost(self, X, y):
        # Pick a random training data point (i), based on the # of training points in X
        i = random.randrange(0, X.shape[0])
        # Returns cost function (MSE) of ((xi * theta) - yi) x[j][i])
        return (X[i].dot(self.theta) - y[i]) * X[i]

    #Cost Function for Mini-Batch GD
