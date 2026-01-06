from Layers import Base
import numpy as np

class BatchNormalization(Base.BaseLayer):

    def __init__(self, channels):
        super().__init__(trainable=True)
        self.channels = channels
        self.initialize()

    def initialize(self): 
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor): 
        # an example shape of input could be  (200, 2, 3, 3) (so batches, channels, rows, cols)
        #First do the Fully Connected Layer approach
        #calculate X tilde per batch
        # for that, calculate batch mean and std
        mean_batch = np.mean(input_tensor, axis=0)
        std_batch = np.std(input_tensor, axis=0)
        x_tilde = (input_tensor - mean_batch) / np.sqrt(std_batch**2 + 1e-11)

        #  calculate the output for this layer as x_tilde * weights (elementwise) + bias
        output = self.weights * x_tilde + self.bias
        
        return output

    def backward(self, error_tensor): 
        #TODO: calculate gradient w.r.t weights (to update the weights)
        #TODO: calculate gradient w.r.t. input (for propogating loss, use helper function compute_bn_gradients)
      return error_tensor
