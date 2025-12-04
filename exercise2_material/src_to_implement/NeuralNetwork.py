import copy
from typing import Any

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer): 
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer: Any = None
        self.loss_layer: Any = None
        

    def forward(self):
        #get the data and labels from data layer
        input_tensor, self.label_tensor = self.data_layer.next()
        
        for layer in self.layers: #last layer is loss 
            output = layer.forward(input_tensor)
            input_tensor = output
        
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss


    def backward(self):
        # we start from loss layer with label tensor as input
        error_tensor = self.loss_layer.backward(self.label_tensor)
        # propagate backwards through all layers in reverse order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
            

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    

    def train(self, iterations):
        
        for _ in range(iterations): 
            self.loss.append(self.forward())
            self.backward()
            

    def test(self, input_tensor):
        """
        propagates the input tensor through the network and returns the prediction of the last layer. 
        For classification tasks we typically query the probabilistic output of the SoftMax layer.
        """
        for layer in self.layers[:-1]: 
            input_tensor = layer.forward(input_tensor)
        
        softmax_input = input_tensor
        
        # TODO: query the probabilsitic output of sotmax layer (the last layer)
        softmax_output = self.layers[-1].forward(softmax_input)
        
        return softmax_output

