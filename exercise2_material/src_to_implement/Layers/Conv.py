from Layers import Base
import numpy as np

class Conv(Base.BaseLayer):

    def __init__(self, stride_shape: int | tuple, convolution_shape: list, num_kernels: int):
        super().__init__(trainable=True) #this is a trainable layer ofc
        
        self.stride_shape = stride_shape # if tuple, allows for diff strides in the spatial dimensions
        self.convolution_shape = convolution_shape # decides if 1/2D Conv. 1D=[c,m], 2D=[c,m,n]; c:= #input channels, m/n:= width/height
        self.num_kernels = num_kernels # this is called H in the slides
        
        # these need to be inited uniformly random in [0,1) 
        self.weights = np.random.uniform(size=(self.num_kernels, *self.convolution_shape)) # shape depends on H and conv_shape, so shape(weights) = [H, convolution_shape]
        
        print("SHAPE OF CONV SHAPE: /n", self.convolution_shape)
        print("MUM OF KERNEL : /n", self.num_kernels)
        print("SHAPE OF WEIGHTS: /n", self.weights.shape)

        self.bias = np.random.uniform(size=self.num_kernels) #each filter gets one scalar bias

   # these need to be calculated in backward pass 
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

        
    def initialize(self, weights_initializer, bias_initializer):
        """Use the given initializers to init weights and bias"""
        pass

    def forward(self, input_tensor): 
        """
        The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order. Here,
        b stands for the batch, c represents the channels and x, y represent the spatial dimensions.
        """
        #TODO: use zero-padding to align spatial shape of input and output for a stride of 1
        #TODO: Handle 1D Convolutions and 1x1 Convolution properly
        """
        Hint: Using correlation in the forward and convolution/correlation in the backward pass
        might help with the flipping of kernels.
        Hint 2: The scipy package features a n-dimensional convolution/correlation.
        Hint 3: Efficiency trade-offs will be necessary in this scope. For example, striding may
        be implemented wastefully as subsampling after convolution/correlation.
        """
        pass


    def backward(self, error_tensor):
        pass