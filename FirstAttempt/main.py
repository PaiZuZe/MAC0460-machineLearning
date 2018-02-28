import numpy as np

def PLA(input, expected, weights) :
    result = np.sign(np.dot(weights, input))
    if result != expected :
        weights += input*expected
    return weights

def perceptron() :
    n = int(input("How many times will I go through the data ?\n"))
    #Our traning data.
    data = [[[1, 0, 0], -1], [[1, 0, 1], 1], [[1, 1 ,0], 1], [[1, 1, 1], 1]]
    #the weights were predetermined in the exercecise.
    weights = np.array([0.5, -1, 1])

    for j in range(n) :
        for i in data :
            weights = PLA(np.array(i[0]), i[1], weights)

perceptron()
