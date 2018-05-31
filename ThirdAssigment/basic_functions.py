import torch.nn.functional as F
import numpy as np
import torch
from util import randomize_in_place


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def graph1(a_np, b_np, c_np):
    """
    Computes the graph
        - x = a * c
        - y = a + b
        - f = x / y

    Computes also df/da using
        - Pytorchs's automatic differentiation (auto_grad)
        - user's implementation of the gradient (user_grad)

    :param a_np: input variable a
    :type a_np: np.ndarray(shape=(1,), dtype=float64)
    :param b_np: input variable b
    :type b_np: np.ndarray(shape=(1,), dtype=float64)
    :param c_np: input variable c
    :type c_np: np.ndarray(shape=(1,), dtype=float64)
    :return: f, auto_grad, user_grad
    :rtype: torch.DoubleTensor(shape=[1]),
            torch.DoubleTensor(shape=[1]),
            numpy.float64
    """
    # YOUR CODE HERE:
    raise NotImplementedError("falta completar a função graph1")
    # END YOUR CODE
    return f, auto_grad, user_grad


def graph2(W_np, x_np, b_np):
    """
    Computes the graph
        - u = Wx + b
        - g = sigmoid(u)
        - f = sum(g)

    Computes also df/dW using
        - pytorchs's automatic differentiation (auto_grad)
        - user's own manual differentiation (user_grad)

    F.sigmoid may be useful here

    :param W_np: input variable W
    :type W_np: np.ndarray(shape=(d,d), dtype=float64)
    :param x_np: input variable x
    :type x_np: np.ndarray(shape=(d,1), dtype=float64)
    :param b_np: input variable b
    :type b_np: np.ndarray(shape=(d,1), dtype=float64)
    :return: f, auto_grad, user_grad
    :rtype: torch.DoubleTensor(shape=[1]),
            torch.DoubleTensor(shape=[d, d]),
            np.ndarray(shape=(d,d), dtype=float64)
    """
    # YOUR CODE HERE:
    raise NotImplementedError("falta completar a função graph2")
    # END YOUR CODE
    return f, auto_grad, user_grad


def SGD_with_momentum(X,
                      y,
                      inital_w,
                      iterations,
                      batch_size,
                      learning_rate,
                      momentum):
    """
    Performs batch gradient descent optimization using momentum.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param inital_w: initial weights
    :type inital_w: np.array(shape=(d, 1))
    :param iterations: number of iterations
    :type iterations: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :param learning_rate: learning rate
    :type learning_rate: float
    :param momentum: accelerate parameter
    :type momentum: float
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    # YOUR CODE HERE:
    raise NotImplementedError("falta completar a função SGD_with_momentum")
    # END YOUR CODE

    return w_np, weights_history, cost_history
