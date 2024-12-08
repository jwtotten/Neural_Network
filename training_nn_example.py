
def sigmoid(x: float):
    return 1/(np.exp(-x)+1)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)