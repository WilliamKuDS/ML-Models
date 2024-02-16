import numpy as np
import time

#Initial Attempt with no inital baseline
class LogisticRegression:
    def __init__(self):
        self.theta = None
        self.learning_rate = None
        self.tolerance = 1e-6
        self.epochs = None

    def fit(self, X, y, learning_rate=0.001, steps=10000, model='baseline'):
        start = time.time()
        self.learning_rate = learning_rate
        self.epochs = self.learning_rate / 1000
        # Get the training examples and features
        m, n = X.shape
        # Initialize theta as zero array with the size of features
        self.theta = np.zeros(n)
        prev_cost = float('inf')
        for step in range(steps):
            if model == 'baseline':
                self.baseline_gradient(X, y, m)
                mse = np.mean((y - self.predict(X)) ** 2)
                print('step: ' + str(step) + ', mse: ' + str(mse))
            elif model == 'newton':
                self.newton_gradient(X, y, n)

        print("Time: {}".format(time.time() - start))

    # Batch Gradient Ascent, calculate h_theta(prediction), calculate log-likelihood of theta, then update theta
    def baseline_gradient(self, X, y, m):
        h_theta = self.sigmoid(X)
        gradient = X.T.dot(y - h_theta)
        # Update Theta with Batch Gradient Ascent
        self.theta += self.learning_rate * gradient / m

    def newton_gradient(self, X, y, n):
        self.theta = self.theta + self.learning_rate * self.hessian_matrix() * self.gradient_loglikelihood()

    def sigmoid(self, X):
        return 1/(1+np.exp(-(X.dot(self.theta))))

    def gradient_loglikelihood(self):
        pass

    def hessian_matrix(self):
        pass

    def newton_step(self, t, d_log_likelihood, d2_log_likelihood):
        self.theta[t+1] = self.theta[t] - d_log_likelihood / d2_log_likelihood

    def log_likelihood(self, X, y):
        m = X.shape[0]
        result = 0
        for i in range(m):
            result += y[i] * np.log(self.theta.dot(X[i])) + (1 - y[i]) * np.log((1-self.theta.dot(X[i])))
        return result

    def batch_gradient_ascent(self, learning_rate, d_log_likelihood):
        self.theta += learning_rate * d_log_likelihood

    def d1loglikelihood(self, X, y, i, j):
        return y[i] - self.theta.dot(X[i]) * X[i][j]

    def predict(self, X):
        return np.dot(X, self.theta)

