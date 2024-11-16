# This is an example of a feedforward neural network with multiple nodes
import numpy as np

def sigmoid(x: float):
    return 1/(np.exp(-x)+1)

class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return sigmoid(z)
    

class NeuralNetwork:

    def __init__(self) -> None:
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights=weights, bias=bias)
        self.h2 = Neuron(weights=weights, bias=bias)
        self.o1 = Neuron(weights=weights, bias=bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs of O1 is using the outputs h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1 

        

if __name__ == "__main__":
    network = NeuralNetwork()
    x = np.array([2, 3])

    print(network.feedforward(x))
    