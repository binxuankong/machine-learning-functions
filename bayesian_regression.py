import numpy as np
import matplotlib.pyplot as plt
from answers import *

# Training data
N = 50
# Input
X = np.reshape(np.linspace(0, 1.0, N), (N, 1))
# Output
Y = np.cos(10*X**2) - 0.6 * np.sin(X**2) + 0.5 * X**2

alpha = 1
beta = 0.1

x_start = np.array([0.25, 0.4])
precision = 0.0001
max_iterations = 150

# The basis function
# :param x: input data
def basis_function(x):
    # Form the basis
    means = np.reshape(np.linspace(-0.5, 1.0, 10), (10, 1))
    sigma = 0.1
    temp = []
    for row in x:
        temp_row = []
        for i in range(means.shape[0]):
            gaussian = np.exp(-((row[0] - means[i][0])**2) / (2 * sigma**2))
            temp_row.append(gaussian)
        temp.append(temp_row)
    basis = np.asarray(temp)
    return basis

Phi = basis_function(X)

# The log marginal likelihood
# :param alpha: alpha value
# :param beta:  beta value
# :param Phi:   basis
# :param Y:     output training data
def lml(alpha, beta, Phi, Y):
    t_1 = alpha * np.dot(Phi, Phi.transpose())
    t_1 = t_1 + beta * np.identity(t_1.shape[0])
    t_2 = -0.5 * np.dot(np.dot(Y.transpose(), np.linalg.inv(t_1)), Y)
    t_3 = -0.5 * np.log(np.linalg.det(t_1))
    t_4 = -0.5 * Y.shape[0] * np.log(2 * np.pi)
    return t_2[0][0] + t_3 + t_4

# The gradient of the log marginal likelihood, in terms of alpha and beta
# :param alpha: alpha value
# :param beta:  beta value
# :param Phi:   basis
# :param Y:     output training data
def grad_lml(alpha, beta, Phi, Y):
    K = alpha * np.dot(Phi, Phi.transpose())
    K = K + beta * np.identity(K.shape[0])
    Kinv = np.linalg.inv(K)
    dK_da = np.dot(Phi, Phi.transpose())
    dK_db = np.identity(K.shape[0])
    dKinv_da = - np.dot(np.dot(Kinv, dK_da), Kinv)
    dKinv_db = - np.dot(np.dot(Kinv, dK_db), Kinv)
    t_1a = -0.5 * np.trace(np.dot(Kinv, dK_da))
    t_2a = -0.5 * np.dot(np.dot(Y.transpose(), dKinv_da), Y)
    t_1b = -0.5 * np.trace(np.dot(Kinv, dK_db))
    t_2b = -0.5 * np.dot(np.dot(Y.transpose(), dKinv_db), Y)
    grad_alpha = t_1a + t_2a
    grad_beta = t_1b + t_2b
    return np.array([grad_alpha[0][0], grad_beta[0][0]])

# Return the mean and variance of the data
# :param x: input data
def predict(x):
    S_N = (1 / beta) * np.dot(Phi.transpose(), Phi)
    S_N = S_N + (1 / alpha) * np.identity(S_N.shape[0])
    S_N = np.linalg.inv(S_N)
    m_N = (1 / beta) * np.dot(Phi.transpose(), Y)
    m_N = np.dot(S_N, m_N)
    Phi_x = basis_function(x)
    E = np.dot(Phi_x, m_N)
    V = np.dot(Phi_x, np.dot(S_N, Phi_x.transpose()))
    return E, V

# Get a gaussian normal distribution of the data
# :param x: the input data
def get_sample(x):
    E, V = predict(x)
    s = np.random.multivariate_normal(E.flatten(), V)
    return s

# Return the upper and lower error bar in terms of the standard deviation,
# excluding the noise
# :param E: mean
# :param V: variance
def std_deviation_error_bar(E, V):
    std = np.sqrt(np.diagonal(V))
    std_up = []
    std_down = []
    for i in range(0, E.shape[0]):
        std_up.append(E[i][0] + std[i])
        std_down.append(E[i][0] - std[i])
    return std_up, std_down

# Return the upper and lower error bar in terms of the standard deviation,
# including the noise
# :param E: mean
# :param V: variance
def noise_error_bar(E, V):
    std = np.sqrt(np.diagonal(V))
    error_up = []
    error_down = []
    for i in range(0, E.shape[0]):
        error_up.append(E[i][0] + std[i] + beta)
        error_down.append(E[i][0] - std[i] - beta)
    return error_up, error_down

# Plot the graph of gaussian regression
def gaussian_plot():
    # Test data
    x = np.reshape(np.linspace(-1.0, 2.0, 250), (250, 1))
    E, V = predict(x)
    sample1 = get_sample(x)
    sample2 = get_sample(x)
    sample3 = get_sample(x)
    sample4 = get_sample(x)
    sample5 = get_sample(x)
    error_bar_up, error_bar_down = std_deviation_error_bar(E, V)
    noise_bar_up, noise_bar_down = noise_error_bar(E, V)
    plt.figure
    plt.plot(x, sample1, 'r-', label='sample 1')
    plt.plot(x, sample2, 'y-', label='sample 2')
    plt.plot(x, sample3, 'b-', label='sample 3')
    plt.plot(x, sample4, 'g-', label='sample 4')
    plt.plot(x, sample5, 'm-', label='sample 5')
    plt.plot(x, E, 'k-', label='predictive mean')
    plt.plot([0], [0], 'c-', label='standard deviation error bar')
    plt.fill_between(x.flatten(), error_bar_up, error_bar_down, facecolor='cyan', alpha=0.3)
    plt.plot(x, noise_bar_up, 'k:', label='error bar with noise')
    plt.plot(x, noise_bar_down, 'k:')
    plt.plot(X, Y, 'ko', label='data')
    plt.xlim(-1.0, 1.5)
    plt.title('Predicted Function Values and Predictive Mean using Gaussian Basis Function', fontsize=20)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    leg = plt.legend(fontsize=12)
    leg.get_frame().set_edgecolor('black')
    plt.show()