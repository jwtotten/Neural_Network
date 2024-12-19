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

    def iterate_region(self, image):
        '''
        Generate all possible image 3x3 regions with padding.
        :param image: The image to create calculate all 3x3 regions in.
        :type image: image
        '''
        height, width = image.shape

        for i in range(0, height-2):
            for j in range(0, width-2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    
    def forward(self, input_image):
        '''
        Performs a forward pass of the convolution layer.
        :param input_image: 2D numpy array
        :type input_image: numpy.array
        '''

        height, width = input_image.shape

        output = np.zeros((height - 2, width - 2, self.num_filters))

        for img_region, i, j in self.iterate_region(input_image):
            output[i, j] = np.sum(img_region * self.filters, axis=(1, 2))

        return output


