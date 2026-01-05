import numpy as np
class Sgd: 
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        """returns the updated weights according to the basic gradient descent update scheme"""
        
        # we already have the gradient_tensor so we just need to subtract the gradients scaled with lr from the weights
        weight_tensor = np.array(weight_tensor)
        gradient_tensor = np.array(gradient_tensor)
        
        return weight_tensor - (self.learning_rate * gradient_tensor) 


class SgdWithMomentum: 
    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum = None
    
    def calculate_update(self, weight_tensor, gradient_tensor): 

        weight_tensor = np.array(weight_tensor)
        gradient_tensor = np.array(gradient_tensor)
        # if momentum is not inited yet, do it
        if self.momentum is None: 
            self.momentum = np.zeros_like(weight_tensor)

        # update momentum
        self.momentum = self.momentum_rate * self.momentum - self.learning_rate * gradient_tensor
        # update weights and return them 
        return self.momentum + weight_tensor 
        

class Adam: 
    def __init__(self,learning_rate, mu, rho):
        
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.first_momentum = None
        self.second_momentum = None
        self.iteration = 0

    def calculate_update(self, weight_tensor, gradient_tensor): 

        weight_tensor = np.array(weight_tensor)
        gradient_tensor = np.array(gradient_tensor)
        self.iteration += 1        

        # we only need to check one of them as they are always used together
        if self.first_momentum is None or self.second_momentum is None: 
            self.first_momentum = np.zeros_like(weight_tensor)
            self.second_momentum = np.zeros_like(weight_tensor)
        
        self.first_momentum = self.mu * self.first_momentum + (1-self.mu) * gradient_tensor # mu controls how much we remember and how much emphasis we give to current gradient
        self.second_momentum = self.rho * self.second_momentum + (1-self.rho) * (gradient_tensor * gradient_tensor) # stores the magnitude of the gradient
        # we correct the 0 shifted bias
        corrected_first = self.first_momentum / (1 - pow(self.mu, self.iteration)) 
        corrected_second = self.second_momentum / (1 - pow(self.rho, self.iteration))
        
        # ratio of velocity and magnitude of gradient change
        return weight_tensor - self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

