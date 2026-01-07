from Layers import Base
import numpy as np
from Layers.Helpers import compute_bn_gradients

moving_avg_decay = 0.8

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
        self.runnning_mu = None
        self.runnning_sigma = None
        self.initialized = False
        self._optimizer = None


    def forward(self, input_tensor): 
        self.input_tensor = input_tensor
        self.mean_batch = np.mean(input_tensor, axis=0)
        self.std_batch = np.std(input_tensor, axis=0)

        if not self.initialized:
            # init running averages with mu and sigma from first batch
            self.runnning_mu = self.mean_batch
            self.runnning_sigma = self.std_batch
            self.initialized = True

        # update the running average
        assert self.runnning_mu is not None
        assert self.runnning_sigma is not None
        self.runnning_mu = moving_avg_decay * self.runnning_mu + (1 - moving_avg_decay) * self.mean_batch 
        self.runnning_sigma = moving_avg_decay * self.runnning_sigma + (1 - moving_avg_decay) * self.std_batch

        if self.testing_phase: 
            # we want to use mu and sigma from test set, but is 2 expensive to caclulate. Instead we will keep a moving avergae that then can be used during test time
            self.x_tilde = (input_tensor - self.runnning_mu) / (np.sqrt(self.runnning_sigma ** 2 + 1e-11))
            return self.weights * self.x_tilde + self.bias

        #First do the Fully Connected Layer approach
        #calculate X tilde per batch
        # for that, calculate batch mean and std
        # trainiing time
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


