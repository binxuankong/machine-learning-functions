import numpy as np
import matplotlib.pyplot as plt
from regression_methods import *
from scipy.interpolate import spline
plt.style.use('seaborn-white')

# Training data
N = 50
# Input
X = np.reshape(np.linspace(0, 1.0, N), (N, 1))
# Output
Y = np.cos(10*X) - 0.6 * np.sin(X**2) + 0.5 * X**2

# Plot the graph of linear regression
# :param func: linear regression function
def linear_regression_plot(func):
    # Input of testing data
    x = np.reshape(np.linspace(-0.5, 1.5, 200), (200, 1))
    y_0 = func(x, 0, X, Y)
    y_1 = func(x, 1, X, Y)
    y_2 = func(x, 2, X, Y)
    y_3 = func(x, 3, X, Y)
    y_4 = func(x, 4, X, Y)
    y_10 = func(x, 10, X, Y)
    plt.figure
    plt.plot(X, Y, 'ko', label='data')
    plt.plot(x, y_0, 'b-', label='order 0')
    plt.plot(x, y_1, 'g-', label='order 1')
    plt.plot(x, y_2, 'r-', label='order 2')
    plt.plot(x, y_3, 'c-', label='order 3')
    plt.plot(x, y_4, 'm-', label='order 4')
    plt.plot(x, y_10, 'y-', label='order 10')
    plt.xlim(-0.5, 1.5)
    plt.ylim(-10, 10)
    plt.title('Maximum Likelihood Estimation', fontsize=20)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    leg = plt.legend(fontsize=14)
    plt.show()

# Plot the graph of average error against order of polynomial basis
def poly_cross_validation_plot():
    order = []
    error = []
    mle_error = []
    for i in range(0, 11):
        order.append(i)
        error.append(polynomial_validation(i, X, Y))
        mle_error.append(ml_error_estimator(polynomial_regression, i, X, Y))
    order_new = np.linspace(0, 10, 200)
    error_smooth = spline(order, error, order_new)
    mle_error_smooth = spline(order, mle_error, order_new)
    plt.figure
    plt.plot(order, error, 'bo', label='average test error')
    plt.plot(order_new, error_smooth, 'b-')
    plt.plot(order, mle_error, 'r*', label='maximum likelihood value for $σ^2$')
    plt.plot(order_new, mle_error_smooth, 'r-')
    plt.title('Leave-One-Out Cross Validation', fontsize=20)
    plt.xlabel('Order of Basis', fontsize=16)
    plt.xticks(np.arange(0, 11, 1))
    leg = plt.legend(fontsize=14)
    leg.get_frame().set_edgecolor('black')
    plt.show()

# Plot the graph of average error against order of trigonometric basis
def trigo_cross_validation_plot():
    order = []
    error = []
    mle_error = []
    for i in range(0, 21):
        order.append(i)
        error.append(trigonometric_validation(i, X, Y))
        mle_error.append(ml_error_estimator(trigonometric_regression, i, X, Y))
    order_new = np.linspace(0, 20, 200)
    error_smooth = spline(order, error, order_new)
    mle_error_smooth = spline(order, mle_error, order_new)
    plt.figure
    plt.plot(order, error, 'bo', label='average test error')
    plt.plot(order_new, error_smooth, 'b-')
    plt.plot(order, mle_error, 'r*', label='maximum likelihood value for $σ^2$')
    plt.plot(order_new, mle_error_smooth, 'r-')
    plt.title('Leave-One-Out Cross Validation', fontsize=20)
    plt.xlabel('Order of Basis', fontsize=16)
    leg = plt.legend(fontsize=14)
    leg.get_frame().set_edgecolor('black')
    plt.show()