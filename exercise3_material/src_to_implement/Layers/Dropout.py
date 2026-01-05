from Layers import Base
import numpy as np

class Dropout(Base.BaseLayer):
    """We will implement inverted Dropout to avoid having to do adjust anything during inference."""
    def __init__(self, probability):
        super().__init__(trainable=False)
        self.probability = probability


    def forward(self, input_tensor):

        # during test time we do not do anything
        if self.testing_phase: 
            return input_tensor

        # as we are doing inverted Dropout, after we set elements to 0 with prob 1-p we need to multiply the remaining elements by 1/p to avoid multiplying with p during test time
        prob = 1 - self.probability

        # create an array of random numbers betweeen 0 and 1 with same shape as input_tensor
        rand_arr = np.random.rand(*input_tensor.shape)
        # create a mask where only values below p are set to 1
        self.mask = rand_arr < prob

        # numpy arrays are passed as references so we copy it
        input_tensor_copy = np.array(input_tensor)
        # use that mask to set elements of input to 0
        input_tensor_copy[self.mask] = 0

        #inverted Dropout
        input_tensor_copy *= 1/self.probability

        return input_tensor_copy

    def backward(self, error_tensor):

        # the gradient is just mask * error_tensor * 1/p
        # numpy arrays are passed as references so we copy it
        error_copy = np.array(error_tensor)
        error_copy[self.mask] = 0
        error_copy *= 1/self.probability

        return error_copy
