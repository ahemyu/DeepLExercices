import copy
from typing import Any
"""
The Neural Network defines the whole architecture by containing all its layers from the input
to the loss layer. This Network manages the testing and the training, that means it calls all
forward methods passing the data from the beginning to the end, as well as the optimization
by calling all backward passes afterwards.
Task:
Implement a class NeuralNetwork in the file: “NeuralNetwork.py” in the same folder as
“NeuralNetworkTests.py”.
 Implement five member variables. An optimizer object received upon construction as
the first argument. A list loss which will contain the loss value for each iteration after
calling train. A list layers which will hold the architecture, a member data layer, which
will provide input data and labels and a member loss layer referring to the special layer
providing loss and prediction. You do not need to care for filling these members with
actual values. They will be set within the unit tests.
 Implement a method forward using input from the data layer and passing it through
all layers of the network. Note that the data layer provides an input tensor and a
label tensor upon calling next() on it. The output of this function should be the
output of the last layer (i. e. the loss layer) of the network.
 Implement a method backward starting from the loss layer passing it the label tensor
for the current input and propagating it back through the network.
 Implement the method append layer(layer). If the layer is trainable, it makes a
deep copy of the neural network’s optimizer and sets it for the layer by using its
optimizer property. Both, trainable and non-trainable layers, are then appended to the
list layers.
Note: We will implement optimizers that have an internal state in the upcoming exercises,
which makes copying of the optimizer object necessary.
 Additionally implement a convenience method train(iterations), which trains the net-
work for iterations and stores the loss for each iteration.
 Finally implement a convenience method test(input tensor) which propagates the in-
put tensor through the network and returns the prediction of the last layer. For clas-
sification tasks we typically query the probabilistic output of the SoftMax layer.
You can verify your implementation using the provided testsuite by providing the commandline
parameter TestNeuralNetwork1
"""
class NeuralNetwork:

    def __init__(self, optimizer): 
        self.optimizer = optimizer
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