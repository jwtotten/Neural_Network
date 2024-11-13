# This is an example of a feedforward neural network
import numpy as np

def sigmoid(x: float):
    return 1/(np.exp(-x)+1)

class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        z = np.dot(self.weights, input) + self.bias
        return sigmoid(z)
    
if __name__ == "__main__":
    weights = np.array([0, 1])
    bias = 4

    n = Neuron(weights=weights, bias=bias)


    x = np.array([0, 1])
    print(n.feedforward(x))
