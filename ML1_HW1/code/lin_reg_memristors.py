from enum import Enum
from typing import Tuple
import numpy as np


class MemristorFault(Enum):
    IDEAL = 0
    DISCORDANT = 1
    STUCK = 2
    CONCORDANT = 3


def model_to_use_for_fault_classification():
    return 2 # TODO: change this to either 1 or 2 (depending on which model you decide to use)


def fit_zero_intercept_lin_model(x: np.ndarray, y: np.ndarray) -> float:
    #"""
    #:param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    #:param y: y coordinates of data points (i.e., \Delta R_i$)
    #:return: theta
    #"""

    # TODO: implement the equation for theta containing sums
    theta = 0
    sum_top = 0
    sum_bottom = 0
    for i in range(len(x)):
            sum_top += y[i] * x[i]
            sum_bottom += x[i] * x[i]
    theta = sum_top / sum_bottom
    return theta


def bonus_fit_lin_model_with_intercept_using_pinv(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    #"""
    #:param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    #:param y: y coordinates of data points (i.e., \Delta R_i$)
    #:return: theta_0, theta_1
    #"""
    from numpy.linalg import pinv

    # TODO: implement the equation for theta using the pseudo-inverse (Bonus Task)
    theta = [None, None]
    return theta[0], theta[1]


def fit_lin_model_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    #"""
    #:param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    #:param y: y coordinates of data points (i.e., \Delta R_i$)
    #:return: theta_0, theta_1
    #"""

    # TODO: implement the equation for theta_0 and theta_1 containing sums
    avg_ideal = x.mean()
    avg_actual = y.mean()
    theta_1_sum_top = 0
    theta_1_sum_top_part1 = 0
    theta_1_sum_top_part2 = 0
    for i in range(len(x)):
        theta_1_sum_top_part1 += y[i] * x[i]
        theta_1_sum_top_part2 += x[i]
    theta_1_sum_top_part2 *= avg_actual
    theta_1_sum_top = theta_1_sum_top_part1 - theta_1_sum_top_part2

    theta_1_sum_bottom = 0
    theta_1_sum_bottom_part1 = 0
    theta_1_sum_bottom_part2 = 0
    for i in range(len(x)):
        theta_1_sum_bottom_part1 += x[i] * x[i]
        theta_1_sum_bottom_part2 += x[i]
    theta_1_sum_bottom_part2 *= avg_ideal
    theta_1_sum_bottom = theta_1_sum_bottom_part1 - theta_1_sum_bottom_part2


    theta_1 = theta_1_sum_top / theta_1_sum_bottom
    theta_0 = avg_actual - theta_1 * avg_ideal
    return theta_0, theta_1


def classify_memristor_fault_with_model1(theta: float) -> MemristorFault:
    """
    :param theta: the estimated parameter of the zero-intercept linear model
    :return: the type of fault
    """
    # TODO: Implement either this function, or the function `classify_memristor_fault_with_model2`,
    #       depending on which model you decide to use.

    # If you decide to use this function, remove the line `raise NotImplementedError()` and
    # return a MemristorFault based on the value of theta.
    # For example, return MemristorFault.IDEAL if you decide that the given theta does not indicate a fault, and so on.
    # Use if-statements and choose thresholds for the parameters that make sense to you.

    raise NotImplementedError()


def classify_memristor_fault_with_model2(theta0: float, theta1: float) -> MemristorFault:
    """
    :param theta0: the intercept parameter of the linear model
    :param theta1: the slope parameter of the linear model
    :return: the type of fault
    """
    # TODO: Implement either this function, or the function `classify_memristor_fault_with_model1`,
    #       depending on which model you decide to use.

    # If you decide to use this function, remove the line `raise NotImplementedError()` and
    # return a MemristorFault based on the value of theta0 and theta1.
    # For example, return MemristorFault.IDEAL if you decide that the given theta pair
    # does not indicate a fault, and so on.
    # Use if-statements and choose thresholds for the parameters that make sense to you.

    def IsBetween(low, num, high):
        if low < num < high:
            return True
        else:
            return False
    threshold_theta0 = 50
    threshold_theta1 = 0.15
    # ideal memristor       theta_1 = 1
    if (IsBetween(1 - threshold_theta1, theta1, 1 + threshold_theta1) and IsBetween(-threshold_theta0, theta0, threshold_theta0)):
        return MemristorFault.IDEAL
    # stuck fault           theta_1 = 0
    if IsBetween(0 - threshold_theta1, theta1, 0 + threshold_theta1):
        return MemristorFault.STUCK
    # discordant fault      theta_1 < 0
    if theta1 < 0:
        return MemristorFault.DISCORDANT
    # concordant fault      theta_1 > 0
    if theta1 > 0:
        return MemristorFault.CONCORDANT
    #raise NotImplementedError()
