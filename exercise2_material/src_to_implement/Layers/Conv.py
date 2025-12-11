from Layers import Base
import numpy as np
from scipy import signal
import math

class Conv(Base.BaseLayer):

    def __init__(self, stride_shape: int | tuple, convolution_shape: list, num_kernels: int):
        super().__init__(trainable=True) #this is a trainable layer ofc
        
        self.stride_shape = stride_shape # if tuple, allows for diff strides in the spatial dimensions, so either n or (m, n); m for height and n for width (only for 2d input ofc)
        self.convolution_shape = convolution_shape # decides if 1/2D Conv. 1D=[c,m], 2D=[c,m,n]; c:= #input channels, m/n:= width/height
        self.num_kernels = num_kernels # this is called H in the slides
        
        # these need to be inited uniformly random in [0,1) 
        self.weights = np.random.uniform(size=(self.num_kernels, *self.convolution_shape)) # shape depends on H and conv_shape, so shape(weights) = [H, convolution_shape]
        
        # print("SHAPE OF CONV SHAPE: /n", self.convolution_shape)
        # print("MUM OF KERNEL : /n", self.num_kernels)
        # print("SHAPE OF WEIGHTS: /n", self.weights.shape)

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

        print("INPUT_TENSOR SHAPE BEFORE PADDING: /n", input_tensor.shape)



        # this is assuming a stride of 1, diff stride will be handled later after convolutiuon was done
        # determine dimensions and mode
        # convolution_shape is [c, m] (1D) or [c, m, n] (2D)
        is_1d = len(self.convolution_shape) == 2
        
        # calculate padding for y
        kernel_y = self.convolution_shape[1]
        pad_total_y = kernel_y - 1
        
        pad_top = pad_total_y // 2
        pad_bottom = pad_total_y - pad_top # handles asymmetry if total is odd
        
        if is_1d:
            # 1D Input Tensor Shape: (batch, channel, y)
            # We only pad the last dimension (axis 2)
            # pad_width format: ((before_0, after_0), (before_1, after_1), ...)
            pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom))
            
        else:
            # 2D Input Tensor Shape: (batch, channel, y, x)
            # Calculate Padding for x
            kernel_x = self.convolution_shape[2]
            pad_total_x = kernel_x - 1
            
            pad_left = pad_total_x // 2
            pad_right = pad_total_x - pad_left
            
            # We pad Y (axis 2) and X (axis 3)
            pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
            
        #apply 0-Padding
        input_padded = np.pad(input_tensor, pad_width, mode='constant')
        

        # now correlate/convolve the paqdded input
        input_conv = signal.
        print("INPUT_TENSOR SHAPE After PADDING: /n", input_padded.shape)

        

        """
        Hint: Using correlation in the forward and convolution/correlation in the backward pass
        might help with the flipping of kernels.
        Hint 2: The scipy package features a n-dimensional convolution/correlation. use signal.correlate()
        Hint 3: Efficiency trade-offs will be necessary in this scope. For example, striding may
        be implemented wastefully as subsampling after convolution/correlation.
        """
        pass


    def backward(self, error_tensor):
        pass
