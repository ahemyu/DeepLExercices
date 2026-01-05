import numpy as np
from Layers import Base

class Flatten(Base.BaseLayer): 
    """Layer to flatten multidim input to 1 dim"""
    def __init__(self): 
        super().__init__(trainable=False)
    
    def forward(self, input_tensor):
        input_tensor = np.array(input_tensor) # shape: (batch_size, channels, height, width)
        # we want to keep the batch dimension
        # and not swap axes
        # so target shape is (batch_size, channels * height * width)
        self.input_shape = input_tensor.shape
        batch_size = self.input_shape[0]   
        return np.reshape(input_tensor, (batch_size, -1))
        
    
    def backward(self, error_tensor): 
        error_tensor = np.array(error_tensor)
        # this has shape (batch_size, channels * height * width)
        # as we saved the shape of input, we can recover it
        batch_size, channels, height, width = self.input_shape
        
        return np.reshape(error_tensor, (batch_size, channels, height, width))
