import numpy as np
from Layers import Base
"""
The SoftMax activation function is used to transform the logits (the output of the network)
into a probability distribution. Therefore, SoftMax is typically used for classification tasks.
Task:
Implement a class SoftMax in the file: “SoftMax.py” in folder “Layers”. This class also has
to provide the methods forward(input tensor) and backward(error tensor).
 Write a constructor for this class, receiving no arguments.
 Implement a method forward(input tensor) which returns the estimated class proba-
bilities for each row representing an element of the batch.
 Implement a method backward(error tensor) which returns a tensor that serves as
the error tensor for the previous layer.
Hint: again the same hint as before applies.
 Remember: Loops are slow in Python. Use NumPy functions instead!
You can verify your implementation using the provided testsuite by providing the commandline
parameter TestSoftMax
"""
class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor): # dimension: (batch_size x #classes)

        self._input_tensor = input_tensor #to be used in backward pass 
        #TODO: for numerical stability we need the max of each row 
        batch_maximas = np.max(input_tensor, axis=1, keepdims=True) 
        shifted_input = input_tensor - batch_maximas # numerical stability goes brr
        
        exp_input = np.exp(shifted_input) # exponential of each element
        exp_sum_rows = np.sum(exp_input, axis=1, keepdims=True) # array of exp sums per batch

        self._probs = np.divide(exp_input, exp_sum_rows)
        
        return self._probs


    def backward(self, error_tensor): 
        # TODO: first we need to multiply error_tensor with probs element wise and sum it per row(axis = 1)
        total_weighted_error = np.sum(error_tensor * self._probs, axis=1, keepdims=True)
        net_gradient_magnitude = error_tensor - total_weighted_error
        
        scaled_result = self._probs * net_gradient_magnitude
        
        return scaled_result
    