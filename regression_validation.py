import numpy as np
from regression_methods import *

# Leave-one-out cross validation (using polynomial basis function)
# :param n: order of basis
# :param X: input of training data
# :param Y: output of training data
def polynomial_validation(n, X, Y):
    error = 0
    for i in range(X.shape[0]):
        X_in = np.delete(X, i, 0)
        Y_in = np.delete(Y, i, 0)
        x_out = np.take(X, i)
        true_y = np.take(Y, i)
        # Form the basis
        temp = []
        for row in X_in:
            temp_row = [1]
            for i in range(1, n+1):
                temp_row.append(row[0]**i)
            temp.append(temp_row)
        basis = np.asarray(temp)
        # Derive the MLE parameters
        t_1 = np.matmul(basis.transpose(), basis)
        t_2 = np.linalg.inv(t_1)
        t_3 = np.matmul(t_2, basis.transpose())
        MLE = np.dot(t_3, Y_in)
        # Predict
        y_out = 0
        for i in range(MLE.shape[0]):
            y_out += MLE[i][0] * row[0]**i
        error += (true_y - y_out)**2
    return error/X.shape[0]

# Leave-one-out cross validation (using trigonometric basis function)
# :param n: order of basis
# :param X: input of training data
# :param Y: output of training data
def trigonometric_validation(n, X, Y):
    error = 0
    for i in range(X.shape[0]):
        X_in = np.delete(X, i, 0)
        Y_in = np.delete(Y, i, 0)
        x_out = np.take(X, i)
        true_y = np.take(Y, i)
        # Form the basis
        temp = []
        for row in X_in:
            temp_row = [1]
            for j in range(1, n+1):
                temp_row.append(np.sin(2*np.pi*j*row[0]))
                temp_row.append(np.cos(2*np.pi*j*row[0]))
            temp.append(temp_row)
        basis = np.asarray(temp)
        # Derive the MLE parameters
        t_1 = np.matmul(basis.transpose(), basis)
        t_2 = np.linalg.inv(t_1)
        t_3 = np.matmul(t_2, basis.transpose())
        MLE = np.dot(t_3, Y_in)
        # Predict
        temp_row = [1]
        for j in range(1, n+1):
            temp_row.append(np.sin(2*np.pi*j*x_out))
            temp_row.append(np.cos(2*np.pi*j*x_out))
        temp = np.asarray(temp_row)
        y_out = np.dot(MLE.transpose(), temp)[0]
        error += (true_y - y_out)**2
    return error/X.shape[0]

# Leave-one-out cross validation (using gaussian basis function)
# :param n: number of occurences of mean = n * 10
# :param X: input of training data
# :param Y: output of training data
def gaussian_regress_validation(n, X, Y):
    error = 0
    for i in range(X.shape[0]):
        X_in = np.delete(X, i, 0)
        Y_in = np.delete(Y, i, 0)
        x_out = np.take(X, i)
        true_y = np.take(Y, i)
        # Form the basis
        means = np.reshape(np.linspace(0, 1, n*10), (n*10, 1))
        sigma = 0.1
        temp = []
        for row in X:
            temp_row = []
            for i in range(means.shape[0]):
                gaussian = np.exp(-((row[0] - means[i][0])**2) / (2 * sigma**2))
                temp_row.append(gaussian)
            temp.append(temp_row)
        basis = np.asarray(temp)
        # Derive the MLE parameters
        t_1 = np.matmul(basis.transpose(), basis)
        t_2 = np.linalg.inv(t_1)
        t_3 = np.matmul(t_2, basis.transpose())
        MLE = np.dot(t_3, Y_in)        # Predict
        temp_row = []
        for i in range(MAP.shape[0]):
            gaussian = np.exp(-((row[0] - means[i][0])**2) / (2 * sigma**2))
            temp_row.append(gaussian)
        temp = np.asarray(temp_row)
        y_out = np.dot(MLE.transpose(), temp)[0]
        error += (true_y - y_out)**2
    return error/X.shape[0]

# Estimate the maximum likelihood value for σ^2
# :param func: linear regression function
# :param n: order of basis
# :param X: input of training data
# :param Y: output of training data
def ml_error_estimator(func, n, X, Y):
	error = 0
	predict = func(X, n, X, Y)
	for i in range(Y.shape[0]):
		error += (Y[i][0] - predict[i][0])**2
	return error/Y.shape[0]