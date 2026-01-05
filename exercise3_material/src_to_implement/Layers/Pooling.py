from Layers import Base
import numpy as np
import math

class Pooling(Base.BaseLayer): 
    """Implementation of Max Pooling Layer"""
    def __init__(self, stride_shape, pooling_shape):
        super().__init__(trainable=False)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def _get_slice_bounds(self, ky, kx, output_dim_y, output_dim_x):
        """Calculate slice boundaries for a given kernel offset."""
        # helper method to calculate the start and end indices for slicing the input tensor
        # based on the current kernel position (ky, kx) and the stride
        stride_y, stride_x = self.stride_shape
        
        # the start index is simply the current kernel offset
        y_start = ky
        # the end index is calculated to ensure we cover enough pixels to match the output dimension
        # when stepping by stride_y
        y_end = ky + output_dim_y * stride_y
        
        x_start = kx
        x_end = kx + output_dim_x * stride_x
        
        return y_start, y_end, x_start, x_end

    def forward(self, input_tensor):
        """
        Different to the convolutional layer, the pooling layer must be implemented only for the 2D case.
        Use “valid”-padding for the pooling layer. This means, unlike to the convolutional layer, don't apply any “zero”-padding. 
        This may discard border elements of the input tensor. 
        Take it into account when creating your output tensor.
        """
        # store input tensor for use in backward pass to determine where max values came from
        self.input_tensor = input_tensor
        
        # unpack shapes for readability
        batch_size, num_channels, h_in, w_in = input_tensor.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        # determine shape of output_tensor using "valid" padding logic
        # valid padding means we only compute the pooling if the kernel fits entirely inside the image
        # mathematically, this corresponds to floor((input - kernel) / stride) + 1
        # any fractional part represents border pixels that are discarded
        output_dim_y = math.floor((h_in - pool_y) / stride_y) + 1
        output_dim_x = math.floor((w_in - pool_x) / stride_x) + 1
        
        # initialize output tensor with negative infinity
        # this ensures that any pixel value from the image will overwrite the initialization
        self.output_tensor = np.full((batch_size, num_channels, output_dim_y, output_dim_x), -np.inf)
        
        # initialize the winning mask with zeros
        # this mask will eventually store a 1 at every input location that was a "winner" (max value)
        self.winning_mask = np.zeros_like(input_tensor) 

        # first pass: find the maximum values
        # instead of iterating over every output pixel (which would require 4 loops: batch, channel, y, x),
        # we iterate over the kernel dimensions (ky, kx).
        # since pooling kernels are small (e.g. 2x2), this loop runs very few times.
        for ky in range(pool_y):
            for kx in range(pool_x):
                # calculate the slice boundaries for the current kernel offset
                y_start, y_end, x_start, x_end = self._get_slice_bounds(ky, kx, output_dim_y, output_dim_x)
                
                # extract a view of the input corresponding to this kernel position
                # by using 'stride' as the step in the slice (start:end:step), we efficiently 
                # select the pixels for this kernel offset across the entire batch and spatial grid at once.
                # this implicitly handles the sliding window logic without explicit spatial loops.
                input_slice = input_tensor[:, :, y_start:y_end:stride_y, x_start:x_end:stride_x]
                
                # compare the current slice against the max values found so far
                # numpy's maximum function broadcasts efficiently over the batch and channel dimensions
                self.output_tensor = np.maximum(self.output_tensor, input_slice)

        # second pass: populate the winning mask
        # we need a second pass because we first needed to know the true global maximum for each window
        # before we could decide which pixels are winners.
        for ky in range(pool_y):
            for kx in range(pool_x):
                # get the same slice boundaries as in the first pass
                y_start, y_end, x_start, x_end = self._get_slice_bounds(ky, kx, output_dim_y, output_dim_x)
                
                # extract the input slice again
                input_slice = input_tensor[:, :, y_start:y_end:stride_y, x_start:x_end:stride_x]
                
                # check if the pixels in this slice match the final maximum values in output_tensor
                # this creates a boolean mask where true means "this pixel is the max for this window"
                is_winner = (input_slice == self.output_tensor)
                
                # add these winners to the global mask
                # using += is crucial here because of overlapping windows (stride < kernel size).
                # a single input pixel might be the maximum for multiple sliding windows.
                # if so, it should accumulate multiple "wins" (and later multiple gradients).
                self.winning_mask[:, :, y_start:y_end:stride_y, x_start:x_end:stride_x] += is_winner

        return self.output_tensor

    def backward(self, error_tensor):
        """
        Layer has no trainable parameters, hence only gradient with respect to input required
        • We need the stored maxima locations
        • The error is routed towards these locations and is zero for all other pixels
        • In cases where the stride smaller than the kernel size the error might be routed multiple times to the same location and therefore has to be summed up
        """
        # recover shapes for pooling and stride
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        
        # retrieve output dimensions directly from the stored output tensor
        # this guarantees dimensions match the error_tensor provided by the next layer
        _, _, output_dim_y, output_dim_x = self.output_tensor.shape

        # initialize the gradient for the previous layer (same shape as input)
        # this starts as zeros because non-max pixels receive zero gradient
        error_tensor_prev = np.zeros(self.input_tensor.shape)

        # we iterate over the kernel pixels to distribute the incoming gradients
        # this mirrors the logic in forward pass, ensuring we visit every window position
        for ky in range(pool_y):
            for kx in range(pool_x):
                # calculate slice boundaries to map output gradient to input locations
                y_start, y_end, x_start, x_end = self._get_slice_bounds(ky, kx, output_dim_y, output_dim_x)
                
                # extract the view of the input tensor again
                input_slice = self.input_tensor[:, :, y_start:y_end:stride_y, x_start:x_end:stride_x]
                
                # create a mask for the current window position
                # we check: is the pixel at this position equal to the max value stored in output_tensor?
                # if yes, this pixel was the "winner" and deserves to receive the gradient
                is_max_mask = (input_slice == self.output_tensor)
                
                # distribute the gradient
                # we take the incoming error_tensor and mask it so only winning positions get the value
                # we use += to accumulate gradients. if a pixel was the max for multiple overlapping windows,
                # it receives the sum of the gradients from all those windows.
                error_tensor_prev[:, :, y_start:y_end:stride_y, x_start:x_end:stride_x] += error_tensor * is_max_mask

        return error_tensor_prev