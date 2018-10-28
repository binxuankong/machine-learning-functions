import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# Starting value of x
x_start = np.array([0, 0])
# Learning rate of the gradient descent
learning_rate = 0.05
# Precision to stop the gradient descent
precision = 0.0001
# Maximum number of iterations
max_iterations = 100

# The function to perform gradient descent on
def function(x1, x2):
    return 4*x1**2 + 4*x2**2 - 2*x1*x2 -x1 - x2

# The gradient of the function
def gradient_function(x):
    B = np.array([[3, -1], [-1, 3]])
    c = np.array([1, 1])
    return (2 * x) + (2 * np.dot(B, x)) - c

def gradient_descent():
    # Intervals of x1 and x2
    x1 = np.linspace(-0.25, 0.5, 100)
    x2 = np.linspace(-0.25, 0.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    # Calculate the values of Y for each x1, x2
    Y = function(X1, X2)
    x1_array = [x_start[0]]
    x2_array = [x_start[1]]
    current = x_start
    iteration = 0
    difference = 1
    while difference > precision and iteration < max_iterations:
        previous = current
        current = current - learning_rate * gradient_function(previous)
        difference = np.linalg.norm(current - previous)
        iteration = iteration + 1
        x1_array.append(current[0])
        x2_array.append(current[1])
    print("Minimum value is", function(current[0], current[1]), "at (", current[0], ",", current[1], ")")
    plt.figure
    plt.contour(X1, X2, Y, 20, colors='black')
    plt.plot(x1_array, x2_array, 'bo', label='gradient descent step')
    plt.title('Gradient Descent Algorithm', fontsize=24)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    leg = plt.legend(fontsize=14)
    plt.show()