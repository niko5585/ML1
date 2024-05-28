import numpy as np

np.random.seed(10)
def gradient_descent(f, df, x0, y0, learning_rate, lr_decay, num_iters):
    """
    Find a local minimum of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list.
    The function should return the minimizing argument (x, y) and f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x, y (solution), f_list (array of function values over iterations)
    """
    f_list = np.zeros(num_iters) # Array to store the function values over iterations
    x, y = x0, y0
    # TODO: Implement the gradient descent algorithm with a decaying learning rate
    start = f(x0, y0)
    learning_rate_decayed = learning_rate
    for i in range(num_iters):
        dx, dy = df(x, y)
        new_x = x - learning_rate_decayed * dx
        new_y = y - learning_rate_decayed * dy
        f_list[i] = f(new_x, new_y)
        learning_rate_decayed *= lr_decay
        x = new_x
        y = new_y
    return x, y, f_list


def ackley(x, y):
    """
    Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: f(x, y) where f is the Ackley function
    """
    # TODO: Implement the Ackley function (as specified in the Assignment 1 sheet)
    result = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x)) + np.cos(2 * np.pi * y)) + np.exp(1) + 20
    return result


def gradient_ackley(x, y):
    """
    Compute the gradient of the Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: \nabla f(x, y) where f is the Ackley function
    """
    # TODO: Implement partial derivatives of Ackley function w.r.t. x and y
    u = np.sqrt(0.5 * (x ** 2 + y ** 2))
    v = np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)
    deriv_firstTerm_x = 4 * np.exp(-0.2 * u) * (x/u)
    deriv_secondTerm_x = np.exp(0.5 * v) * np.sin(2 * np.pi * x) * np.pi

    deriv_firstTerm_y = 4 * np.exp(-0.2 * u) * (y/u)
    deriv_secondTerm_y = np.exp(0.5 * v) * np.sin(2 * np.pi * y) * np.pi

    df_dx = deriv_firstTerm_x + deriv_secondTerm_x
    df_dy = deriv_firstTerm_y + deriv_secondTerm_y

    gradient = np.array([df_dx, df_dy])
    return gradient