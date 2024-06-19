import numpy
import numpy as np
from sklearn.base import BaseEstimator


def loss(w, b, C, X, y):
    # TODO: Implement the loss function (Equation 1)
    #       useful methods: np.sum, np.clip
    w_squared_euclidean = np.sum(w ** 2)
    max_sum = np.sum(np.maximum(0, 1 - y * (np.dot(X, w) + b)))
    return 0.5 * w_squared_euclidean + C * max_sum


def grad(w, b, C, X, y):
    # TODO: Implement the gradients of the loss with respect to w and b.
    #       Useful methods: np.sum, np.where, numpy broadcasting
    grad_w = np.zeros(w.shape)
    grad_b = 0
    # calc term in sum
    for i in range(len(X)):
        term = y[i] * (numpy.dot(w, X[i]) + b)
        # check what case
        if term < 1: # incorrectly classified
            grad_w -= y[i] * X[i]
            grad_b -= y[i]

    grad_w = w + C * grad_w
    grad_b = C * grad_b
    return grad_w, grad_b


class LinearSVM(BaseEstimator):
    def __init__(self, C=1, eta=1e-3, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, X, y):
        # convert y such that components are not \in {0, 1}, but \in {-1, 1}
        y = np.where(y == 0, -1, 1)

        # TODO: Initialize self.w and self.b. Does the initialization matter?
        self.w = np.zeros(X.shape[1])
        self.b = 0

        loss_list = []
        eta = self.eta  # starting learning rate
        for j in range(self.max_iter):
            # TODO: Compute the gradients, update self.w and self.b using `eta` as the learning rate.
            #       Compute the loss and add it to loss_list.
            grad_w, grad_b = grad(self.w, self.b, self.C, X, y)
            self.w = self.w - eta * grad_w
            self.b = self.b - eta * grad_b

            loss_list.append(loss(self.w, self.b, self.C, X, y))
            # decaying learning rate
            eta = eta * 0.99

        return loss_list

    def predict(self, X):
        # TODO: Predict class labels of unseen data points on rows of X
        #       NOTE: The output should be a vector of 0s and 1s (*not* -1s and 1s)
        classification = np.dot(X, self.w) + self.b
        classification_sign = numpy.sign(classification)
        classification_sign[classification_sign == -1] = 0
        return classification_sign

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
