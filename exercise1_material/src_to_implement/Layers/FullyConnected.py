from Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__(trainable=True)
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.random((output_size, input_size+1)) # +1 bc of bias
        self.gradient_weights = np.zeros_like(self.weights)
        self._optimizer = None

    @property
    def optimizer(self):
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, value):
        if value is None:
                raise ValueError("You must pass a value!")
        self._optimizer = value
             
            
    def forward(self, input_tensor): 
               
        # we need to append row of 1s to input_tensor bc of bias (input_size many)
        bias_row = np.ones(self.input_size)
        input_tensor = np.hstack([input_tensor, bias_row])
        
        #TODO: not sure if I need to transpose first 
        output_tensor = input_tensor @ self.weights 
       
        assert(output_tensor.shape[1] == self.output_size)     

        return output_tensor
    
    def backward(self, error_tensor): 
        pass
    

"""
Implement a class FullyConnected in the file “FullyConnected.py” in folder “Layers”, that
inherits the base layer that we implemented earlier. This class has to provide the methods
forward(input tensor) and backward(error tensor) as well as the property optimizer.

 Write a constructor for this class, receiving the arguments (input size, output size).
First, call its super-constructor. Set the inherited member trainable to True, as this
layer has trainable parameters. Initialize the weights of this layer uniformly random in
the range [0, 1).

TODO Implement a method forward(input tensor) which returns a tensor that serves as the
input tensor for the next layer. input tensor is a matrix with input size columns
and batch size rows. The batch size represents the number of inputs processed si-
multaneously. The output size is a parameter of the layer specifying the number of
columns of the output.

 Add a setter and getter property optimizer which sets and returns the protected member
optimizer for this layer. Properties offer a pythonic way of realizing getters and setters.
Please get familiar with this concept if you are not aware of it.

 Implement a method backward(error tensor) which returns a tensor that serves as
the error tensor for the previous layer. Quick reminder: in the backward pass we are
going in the other direction as in the forward pass.
Hint: if you discover that you need something here which is no longer available to you,
think about storing it at the appropriate time.

 To be able to test the gradients with respect to the weights: The member for the weights
and biases should be named weights. For future reasons provide a property gradi-
ent weights which returns the gradient with respect to the weights, after they have
been calculated in the backward-pass. These properties are accessed by the unit tests
and are therefore also important to pass the tests!

 Use the method calculate update(weight tensor, gradient tensor) of your opti-
mizer in your backward pass, in order to update your weights. Don't perform an
update if the optimizer is not set.
"""