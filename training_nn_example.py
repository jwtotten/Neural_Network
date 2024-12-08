import numpy as np


def sigmoid(x: float):
    return 1/(np.exp(-x)+1)

def mse_loss(y_true, y_pred):
    """
    MSE - mean square error calculation
    """
    return ((y_true - y_pred)**2).mean()

def deriv_sigmoid(x):
    """
    Derivative of the sigmoid function.
    """
    fx = sigmoid(x)
    return fx * (1 - fx)

class Neural_network:
    """
    neural network with 2 inputs.
    There are two layers, a hidden layer with 2 neurons and an output layer with 1 neuron.
    """
   
    def __init___(self):
        # Adding the weights of the network
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        # Adding the biases of the network
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def feed_forward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    