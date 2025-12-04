import numpy as np
class Sgd: 
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        """returns the updated weights according to the basic gradient descent update scheme"""
        
        
        # TODO: calculate the loss with respect to the weights
        # we already have the gradient_tensor so we just need to subtract the gradients scaled with lr from the weights
        weight_tensor = np.array(weight_tensor)
        gradient_tensor = np.array(gradient_tensor)
        
        return weight_tensor - (self.learning_rate * gradient_tensor) 
