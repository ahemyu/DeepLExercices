from Layers import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(trainable=True)
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.random((input_size+1, output_size)) # +1 bc of bias
        self._gradient_weights = np.zeros_like(self.weights)
        self._optimizer = None

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
             
            
    def forward(self, input_tensor): 
               
        # we need to append col of 1s to input_tensor bc of bias (input_size[0] many)
        bias_col = np.ones((input_tensor.shape[0], 1))  # shape: (batch_size, 1)
        self.input_tensor = np.hstack([input_tensor, bias_col])
        
        output_tensor = self.input_tensor @ self.weights 
       
        assert(output_tensor.shape[1] == self.output_size)     

        return output_tensor
    
    def backward(self, error_tensor):
        # error_tensor is gradient flowing back from the layer above so dL/dOutput
        
        gradient_input = error_tensor @ self.weights.T #Gradient of the loss wrt to input
        gradient_input = gradient_input[:, :-1] # remove last column (bias)
        
        self._gradient_weights = (error_tensor.T @ self.input_tensor).T # gradient of Loss wrt to weights

        if self._optimizer: 
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights) #update weights

        return gradient_input