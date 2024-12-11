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
    
    def train(self, data, all_y_trues):
        """
        Method to train the neural network.
        :param data: [n,2] numpy array where n in the number of samples in the training data.
        :param all_y_trues: numpy array which corresponds to the data in 'data'.
        """
        learning_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                ### Calculating the partial derivatives
                dl_dypred = -2 * (y_true - y_pred)

                # Neuron o1
                dypred_dw5 = h1 * deriv_sigmoid(sum_o1)
                dypred_dw6 = h2 * deriv_sigmoid(sum_o1)
                dypred_db3 = deriv_sigmoid(sum_o1)

                dypred_dh1 = self.w5 * deriv_sigmoid(sum_o1)
                dypred_dh2 = self.w6 * deriv_sigmoid(sum_o1)


                # Neuron h1
                dh1_dw1 = x[0] * deriv_sigmoid(sum_h1)
                dh1_dw2 = x[1] * deriv_sigmoid(sum_h1)
                dh1_db1 = deriv_sigmoid(sum_h1)
                
                # Neuron h2
                dh2_dw3 = x[0] * deriv_sigmoid(sum_h2)
                dh2_dw4 = x[1] * deriv_sigmoid(sum_h2)
                dh2_db2 = deriv_sigmoid(sum_h2)

                ### Updating the weights and biases

                # Neuron h1
                self.w1 -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw1
                self.w2 -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw2
                self.b1 -= learning_rate * dl_dypred * dypred_dh1 * dh1_db1

                

                
                                
