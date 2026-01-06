from Layers import Base
import numpy as np
from Layers.Helpers import compute_bn_gradients

class BatchNormalization(Base.BaseLayer):

    def __init__(self, channels):
        super().__init__(trainable=True)
        self.channels = channels
        self.initialize()


    @property
    def optimizer(self):
        return self._optimizer
        

    @optimizer.setter
    def optimizer(self, value):
        if value is None:
                raise ValueError("You must pass a value!")
        self._optimizer = value
        

    @property
    def gradient_weights(self):
        return self._gradient_weights


    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value


    @property
    def gradient_bias(self):
        return self._gradient_bias


    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value


    def initialize(self): 
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self._gradient_weights = np.ones_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self._optimizer = None


    def forward(self, input_tensor): 
        # an example shape of input could be  (200, 2, 3, 3) (so batches, channels, rows, cols)
        #First do the Fully Connected Layer approach
        #calculate X tilde per batch
        # for that, calculate batch mean and std
        self.input_tensor = input_tensor
        self.mean_batch = np.mean(input_tensor, axis=0)
        self.std_batch = np.std(input_tensor, axis=0)
        self.x_tilde = (input_tensor - self.mean_batch) / np.sqrt(self.std_batch**2 + 1e-11)

        #  calculate the output for this layer as x_tilde * weights (elementwise) + bias
        output = self.weights * self.x_tilde + self.bias
        
        return output


    def backward(self, error_tensor): 

        # calculate gradient w.r.t. input (for propogating loss, use helper function compute_bn_gradients)
        gradient_input = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean_batch, self.std_batch ** 2)

        self.gradient_weights = np.sum(error_tensor * self.x_tilde, axis=0)
        # also the bias ofc
        self.gradient_bias = np.sum(error_tensor, axis=0)

        #Upddate weights and bias using the optimizer if present
        if self._optimizer:
            # optimize weights and bias separately
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)


        return gradient_input


