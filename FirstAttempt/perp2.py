import numpy as np

class learning() :
    def __init__(self, input, expected = None) :
        self.W = np.array([0 for i in input[0]])
        self.input = input
        if expected != None:
            self.expected = expected

    def PLA(self, n) :
        for i in range(n) :
            for j in range(len(input)) :
                result = np.sign(np.dot(self.W, self.input[j]))
                if result != self.expected[j] :
                    self.W += np.array(self.input[j])*self.expected[j]

input = [[1, 0, 0], [1, 0, 1], [1, 1 ,0], [1, 1, 1]]
output = [-1, 1, 1, 1]

bob = learning(np.array(input), np.array(output))

bob.PLA(5)
bob.p_weithgs()
