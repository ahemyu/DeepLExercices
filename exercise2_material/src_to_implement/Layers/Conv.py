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
        
        self.bias = np.random.uniform(size=self.num_kernels) #each filter gets one scalar bias
        self._optimizer = None

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

    @property
    def optimizer(self):
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, value):
        if value is None:
                raise ValueError("You must pass a value!")
        self._optimizer = value

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
        # fan_in = c * m * n
        # fan_out = num_kernels * m * n
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        
        weights_shape = (self.num_kernels, *self.convolution_shape)
        self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

    def forward(self, input_tensor): 
        """
        The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order. Here,
        b stands for the batch, c represents the channels and x, y represent the spatial dimensions.
        """

        # for backward pass
        self.input_tensor = input_tensor
        # this is assuming a stride of 1, diff stride will be handled later after convolutiuon was done
        # convolution_shape is [c, m] (1D) or [c, m, n] (2D)
        batch = input_tensor.shape[0]
        self.is_1d = len(self.convolution_shape) == 2
        
        kernel_y = self.convolution_shape[1]
        pad_total_y = kernel_y - 1
        
        pad_top = pad_total_y // 2
        pad_bottom = pad_total_y - pad_top # handles asymmetry if total is odd
        
        if self.is_1d:
            # 1D Input Tensor Shape: (batch, channel, y)
            # We only pad the last dimension
            self.pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom))
         
        else:
            # 2D Input Tensor Shape: (batch, channel, y, x)
            kernel_x = self.convolution_shape[2]
            pad_total_x = kernel_x - 1
            
            pad_left = pad_total_x // 2
            pad_right = pad_total_x - pad_left
            
            self.pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
            
        #apply 0-Padding
        self.input_padded = np.pad(input_tensor, self.pad_width, mode='constant')

        # now we iterate over all batches and then over all kernels to calculate each output feature map
        output_tensor = None

        if self.is_1d:
            # we only have one dimension to take care of so stride also needs to be an int
            if isinstance(self.stride_shape,tuple) or isinstance(self.stride_shape, list):
                self.stride_shape = self.stride_shape[0]
            output_dim = math.ceil(input_tensor.shape[2]/self.stride_shape)
            # dim is batch, num_kernels, output_dim
            output_tensor = np.zeros(shape=(batch, self.num_kernels, output_dim))

        else: 
            # diff dimensions for width and height
            output_dim = (math.ceil(input_tensor.shape[2]/self.stride_shape[0]), math.ceil(input_tensor.shape[3]/self.stride_shape[1]))
            output_tensor = np.zeros(shape=(batch, self.num_kernels, output_dim[0], output_dim[1]))

        for b in range(self.input_padded.shape[0]):
            for k in range(self.num_kernels):

                feature_map = signal.correlate(self.input_padded[b], self.weights[k], mode="valid") #we need to squeeze it bec it returns [1, Y, X]
                feature_map = np.squeeze(feature_map, axis=0)
                feature_map += self.bias[k] # apply bias to output map
                self.feature_map_shape = feature_map.shape

                #slicing as subsampling
                if self.is_1d:
                    output_tensor[b, k, :] = feature_map[::self.stride_shape]
                else:
                    output_tensor[b, k, :, :] = feature_map[::self.stride_shape[0], ::self.stride_shape[1]]

        return output_tensor


    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        num_channels = self.convolution_shape[0]

        #upsample error tensor to undo stride subsampling
        upsampled_shape = (batch_size, self.num_kernels) + self.feature_map_shape
        upsampled_error = np.zeros(upsampled_shape)
        if self.is_1d:
            upsampled_error[:, :, ::self.stride_shape] = error_tensor
        else:
            upsampled_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        #gradient bias: sum over batch and spatial dims
        spatial_axes = tuple(range(2, upsampled_error.ndim))
        self._gradient_bias = np.sum(upsampled_error, axis=(0,) + spatial_axes)

        #gradient weights: correlate padded input with upsampled error
        self._gradient_weights = np.zeros_like(self.weights)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    corr = signal.correlate(self.input_padded[b, c], upsampled_error[b, k], mode='valid')
                    self._gradient_weights[k, c] += corr

        #gradient input: convolve upsampled error with weights (flipped)
        gradient_input_padded = np.zeros_like(self.input_padded)
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(num_channels):
                    grad_contrib = signal.convolve(upsampled_error[b, k], self.weights[k, c], mode='full')
                    gradient_input_padded[b, c] += grad_contrib

        #update weights if optimizer is set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)

        #remove padding to match original input shape and return gradient_input
        if self.is_1d:
            pad_top, pad_bottom = self.pad_width[2]
            if pad_bottom == 0:
                return gradient_input_padded[:, :, pad_top:]
            return gradient_input_padded[:, :, pad_top:-pad_bottom]

        else:
            pad_top, pad_bottom = self.pad_width[2]
            pad_left, pad_right = self.pad_width[3]
            y_end = None if pad_bottom == 0 else -pad_bottom
            x_end = None if pad_right == 0 else -pad_right
            return gradient_input_padded[:, :, pad_top:y_end, pad_left:x_end]
