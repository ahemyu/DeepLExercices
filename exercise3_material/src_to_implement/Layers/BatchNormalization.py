from Layers import Base
import numpy as np

class BatchNormalization(Base.BaseLayer):

    def __init__(self, channels):
        super().__init__(trainable=True)
        self.channels = channels
        self.initialize()

    def initialize(self): 
        self.weights = np.zeros(self.channels)
        self.bias = np.ones(self.channels)

    def forward(self, input_tensor): 
        return input_tensor

    def backward(self, error_tensor): 
      return error_tensor
