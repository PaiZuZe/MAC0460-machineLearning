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
    a = torch.from_numpy(a_np)
    b = torch.from_numpy(b_np)
    c = torch.from_numpy(c_np)
    a.requires_grad = True
    x = a * c
    y = a + b
    f = x / y
    f.backward()
    auto_grad = a.grad
    """
    df/da = df/dx * dx/da + df/dy * dy/da
    df/dx = 1/y, dx/da = c,  df/dy = - x/y², dy/da = 1 
    """
    user_grad = ((c / y) - (x / (y * y))).detach().numpy()
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
    W = torch.from_numpy(W_np)
    W.requires_grad = True
    x = torch.from_numpy(x_np)
    b = torch.from_numpy(b_np)
    u = torch.matmul(W, x) + b
    g = F.sigmoid(u)
    f = torch.sum(g)
    f.backward()
    auto_grad = W.grad
    """
    df_du = sigmoid(u) * (1 - sigmoid(u))
    du_dW = x^t
    df_dW = df_du * du_dW
    """
    xt = torch.transpose(x, 0, 1)
    sigU = F.sigmoid(u)
    user_grad = (torch.matmul((sigU * (1 - sigU)), xt)).detach().numpy()
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
    z = torch.autograd.Variable(torch.zeros(inital_w.shape).double() , requires_grad = True)
    W = torch.autograd.Variable(torch.from_numpy(inital_w), requires_grad = True)
    x = torch.autograd.Variable(torch.from_numpy(X), requires_grad = False)
    Y = torch.autograd.Variable(torch.from_numpy(y), requires_grad = False)    
    cost_history = []
    weights_history = []
    for i in range(iterations) :
        temp = torch.randperm(x.shape[0])
        x = x[temp]
        Y = Y[temp]
        xW = torch.matmul(x[:batch_size], W)
        xWY = xW - Y[:batch_size]
        J = torch.matmul(torch.transpose(xWY, 0, 1), xWY)  / batch_size
        J.backward()
        z.data = momentum * z + W.grad
        W.grad.zero_()
        W.data -= learning_rate * z
        cost_history.append(J)
        weights_history.append(W.data)
    w_np = W.detach().numpy()
    # END YOUR CODE

    return w_np, weights_history, cost_history
