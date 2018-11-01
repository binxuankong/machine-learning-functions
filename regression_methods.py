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

# Linear regression using gaussian basis functions
# :param x: input data
# :param n: number of occurences of mean = n * 10
# :param X: input of training data
# :param Y: output of training data
 def gaussian_regression(x, n, X, Y):
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
    MLE = np.dot(t_3, Y)
    # Predict
    predict= []
    for row in x:
        temp_row = []
        for i in range(MAP.shape[0]):
            gaussian = np.exp(-((row[0] - means[i][0])**2) / (2 * sigma**2))
            temp_row.append(gaussian)
        temp = np.asarray(temp_row)
        y = np.dot(MLE.transpose(), temp)
        predict.append(y)
    np_predict = np.asarray(predict)
    return np_predict