import numpy as np

# Linear regression using polynomial basis functions
# :param x: input of test data
# :param n: order of basis
# :param X: input of training data
# :param Y: output of training data
def polynomial_regression(x, n, X, Y):
    # Form the basis
    temp = []
    for row in X:
        temp_row = []
        for i in range(n+1):
            temp_row.append(row[0]**i)
        temp.append(temp_row)
    basis = np.asarray(temp)
    # Derive the MLE parameters
    t_1 = np.matmul(basis.transpose(), basis)
    t_2 = np.linalg.inv(t_1)
    t_3 = np.matmul(t_2, basis.transpose())
    MLE = np.dot(t_3, Y)
    # Predict
    predict= []
    for row in x:
        y = 0
        for i in range(MLE.shape[0]):
            y += MLE[i][0] * row[0]**i
        predict.append([y])
    predict = np.asarray(predict)
    return predict

# Linear regression using trigonometric basis functions
# :param x: input data
# :param n: order of basis
# :param X: input of training data
# :param Y: output of training data
def trigonometric_regression(x, n, X, Y):
    # Form the basis
    temp = []
    for row in X:
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
    MLE = np.dot(t_3, Y)
    # Predict
    predict= []
    for row in x:
        temp_row = [1]
        for j in range(1, n+1):
            temp_row.append(np.sin(2*np.pi*j*row[0]))
            temp_row.append(np.cos(2*np.pi*j*row[0]))
        temp = np.asarray(temp_row)
        y = np.dot(MLE.transpose(), temp)
        predict.append(y)
    predict = np.asarray(predict)
    return predict

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