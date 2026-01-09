from Layers import Base
import numpy as np
from Layers.Helpers import compute_bn_gradients

moving_avg_decay = 0.8


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__(trainable=True)
        self.channels = channels
        self.initialize()

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        if value is None:
            raise ValueError("You must pass a value!")
        self._optimizer = value

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

    def initialize(self, weights_initializer=None, bias_initializer=None):
        #TODO: actually use the weights_initializer and bias_initializer
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self._gradient_weights = np.ones_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self.runnning_mu = None
        self.runnning_sigma = None
        self.initialized = False
        self._optimizer = None

    def reformat(self, input_tensor):
        """Turn 2D Input into 4D and vice versa. We do this to reuse the 2d code for 4d input."""

        if len(input_tensor.shape) == 4:
            # 4D to 2D
            # that spatial dimensions M , N can be treated like the batch dimension B
            # we can reshape the B × H × M × N tensor to B × H × M · N
            # because of our format we have to transpose from B × H × M · N to B×M·N×H
            # and afterwards reshape again to have a B · M · N × H tensor
            # reshape from B × H × M × N to B × H × M*N
            # example input shape: (5, 3, 6, 4)
            # expected shape after following operation: (5, 3, 24)
            og_shape = input_tensor.shape
            self.reformat_shape = og_shape  # Store for reverse transformation
            input_tensor = np.reshape(
                input_tensor, (og_shape[0], og_shape[1], og_shape[2] * og_shape[3])
            )
            # transpose from B × H × M · N to B×M·N×H
            # so we expect (5, 24, 3) afterwards
            input_tensor = np.transpose(input_tensor, (0, 2, 1))

            # reshape from B×M·N×H to B · M · N × H
            # expected shape after (5, 72)
            input_tensor = np.reshape(
                input_tensor,
                (input_tensor.shape[0] * input_tensor.shape[1], input_tensor.shape[2]),
            )

        else:
            # 2D to 4D
            B, H, M, N = self.reformat_shape

            # reshape from (B·M·N, H) to (B, M·N, H)
            input_tensor = np.reshape(input_tensor, (B, M * N, H))

            # transpose from (B, M·N, H) to (B, H, M·N)
            input_tensor = np.transpose(input_tensor, (0, 2, 1))

            # reshape from (B, H, M·N) to (B, H, M, N)
            input_tensor = np.reshape(input_tensor, (B, H, M, N))

        return input_tensor

    def forward(self, input_tensor):
        self.is_4d = len(input_tensor.shape) == 4
        if self.is_4d:
            input_tensor = self.reformat(input_tensor)

        self.input_tensor = input_tensor
        self.mean_batch = np.mean(input_tensor, axis=0)
        self.std_batch = np.std(input_tensor, axis=0)

        if not self.initialized:
            # init running averages with mu and sigma from first batch
            self.runnning_mu = self.mean_batch
            self.runnning_sigma = self.std_batch
            self.initialized = True

        if self.testing_phase:
            # we want to use mu and sigma from test set, but is 2 expensive to caclulate. Instead we will keep a moving avergae that then can be used during test time
            assert self.runnning_sigma is not None
            assert self.runnning_mu is not None
            self.x_tilde = (input_tensor - self.runnning_mu) / (
                np.sqrt(self.runnning_sigma**2 + 1e-11)
            )
            output = self.weights * self.x_tilde + self.bias
            if self.is_4d:
                output = self.reformat(output)
            return output

        # update the running average (only during training)
        assert self.runnning_mu is not None
        assert self.runnning_sigma is not None
        self.runnning_mu = (
            moving_avg_decay * self.runnning_mu
            + (1 - moving_avg_decay) * self.mean_batch
        )
        self.runnning_sigma = (
            moving_avg_decay * self.runnning_sigma
            + (1 - moving_avg_decay) * self.std_batch
        )

        # First do the Fully Connected Layer approach
        # calculate X tilde per batch
        # for that, calculate batch mean and std
        # trainiing time
        self.x_tilde = (input_tensor - self.mean_batch) / np.sqrt(
            self.std_batch**2 + 1e-11
        )
        #  calculate the output for this layer as x_tilde * weights (elementwise) + bias
        output = self.weights * self.x_tilde + self.bias

        if self.is_4d:
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        # TODO: if we called reformat in forward we need to take into account here
        if self.is_4d:
            error_tensor = self.reformat(error_tensor)

        # calculate gradient w.r.t. input (for propogating loss, use helper function compute_bn_gradients)
        gradient_input = compute_bn_gradients(
            error_tensor,
            self.input_tensor,
            self.weights,
            self.mean_batch,
            self.std_batch**2,
        )

        self.gradient_weights = np.sum(error_tensor * self.x_tilde, axis=0)
        # also the bias ofc
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Upddate weights and bias using the optimizer if present
        if self._optimizer:
            # optimize weights and bias separately
            self.weights = self._optimizer.calculate_update(
                self.weights, self.gradient_weights
            )
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)

        if self.is_4d:
            gradient_input = self.reformat(gradient_input)

        return gradient_input
