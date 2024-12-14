import numpy as np

class Convolution_layer_3x3:
    '''
    A convolution layer that uses a 3x3 Sobel filter.
    '''

    def __init__(self, num_filters):
        '''
        :param num_filters: number of dimensions of the 3D sobel filter.
        :type num_filters: int
        '''
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9 # Creates an array of 3x3 filters, divided by 9 to reduce the variance.

