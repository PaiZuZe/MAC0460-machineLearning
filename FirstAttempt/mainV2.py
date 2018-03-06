import numpy as np

class learning() :
    def __init__(self, input, expected = None, weights = None) :
        self.input = input
        if weights is None:
            self.W = np.array([0 for i in input[0]])
        if expected is not None:
            self.expected = expected

    def PLA(self, n) :
        for i in range(n) :
            for j in range(len(input)) :
                result = np.sign(np.dot(self.W, self.input[j]))
                if result != self.expected[j] :
                    self.W += np.array(self.input[j])*self.expected[j]

    def linear_regression(self) :
        self.W = np.dot(np.linalg.pinv(self.input), self.expected)

    def err_insample(self) :
        count = 0
        for i in range(len(input)) :
            result = np.sign(np.dot(self.W, self.input[i]))
            if result == self.expected[i] :
                count += 1
        return count/len(input)

input = [[1, 0, 0], [1, 0, 1], [1, 1 ,0], [1, 1, 1]]
output = [-1, 1, 1, 1]

example = learning(np.array(input), np.array(output))
