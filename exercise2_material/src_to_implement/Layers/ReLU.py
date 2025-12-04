import numpy as np
from Layers import Base

class ReLU(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        relu_applied = np.maximum(self._input_tensor, 0)
        
        return relu_applied
    
    def backward(self, error_tensor):
        #the derivative of ReLU is 1 if x>0 and 0 if x<= 0, so we essentially just need a mask around the error_tensor (depending on the values of input_tensor)
        mask = (self._input_tensor > 0).astype(int)
        
        return mask * error_tensor #element-wise mult 
