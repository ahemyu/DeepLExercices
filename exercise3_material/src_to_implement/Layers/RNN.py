import numpy as np
from Layers import Base, FullyConnected, TanH
import copy


class RNN(Base.BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(trainable=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.memorize = False
        self.hidden_state = np.zeros((1, hidden_size))

        # FC layer for hidden state: [x_t, h_{t-1}] -> h_t
        self.fc_hidden = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh = TanH.TanH()

        # FC layer for output: h_t -> y_t
        self.fc_output = FullyConnected.FullyConnected(hidden_size, output_size)

        self._optimizer = None

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_hidden.weights = value

    @property
    def gradient_weights(self):
        return self.fc_hidden.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fc_hidden.gradient_weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.output_size))

        # reset hidden state if not memorizing
        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))

        # store for backward pass
        self.input_tensor = input_tensor
        self.hidden_states = []
        self.tanh_activations = []
        self.fc_hidden_inputs = []

        for t in range(batch_size):
            x_t = input_tensor[t:t+1, :]  # shape (1, input_size)

            # concat input and hidden state
            combined = np.hstack([x_t, self.hidden_state])  # shape (1, input_size + hidden_size)
            self.fc_hidden_inputs.append(combined)

            # compute new hidden state
            fc_out = self.fc_hidden.forward(combined)
            self.hidden_state = self.tanh.forward(fc_out)

            self.hidden_states.append(self.hidden_state.copy())
            self.tanh_activations.append(self.tanh.activation)

            # compute output
            output_tensor[t:t+1, :] = self.fc_output.forward(self.hidden_state)

        return output_tensor

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        gradient_input = np.zeros((batch_size, self.input_size))

        # gradient flowing back through hidden state
        grad_hidden = np.zeros((1, self.hidden_size))

        # accumulate gradients for weights
        accumulated_grad_hidden = np.zeros_like(self.fc_hidden.weights)
        accumulated_grad_output = np.zeros_like(self.fc_output.weights)

        # backward through time
        for t in reversed(range(batch_size)):
            # restore state for this timestep
            self.fc_output.input_tensor = np.hstack([self.hidden_states[t], np.ones((1, 1))])

            # backward through output layer
            grad_from_output = self.fc_output.backward(error_tensor[t:t+1, :])
            accumulated_grad_output += self.fc_output.gradient_weights

            # add gradient from output and from next timestep
            total_grad_hidden = grad_from_output + grad_hidden

            # backward through tanh
            self.tanh.activation = self.tanh_activations[t]
            grad_tanh = self.tanh.backward(total_grad_hidden)

            # backward through fc_hidden
            self.fc_hidden.input_tensor = np.hstack([self.fc_hidden_inputs[t], np.ones((1, 1))])
            grad_combined = self.fc_hidden.backward(grad_tanh)
            accumulated_grad_hidden += self.fc_hidden.gradient_weights

            # split gradient for input and previous hidden state
            gradient_input[t, :] = grad_combined[:, :self.input_size]
            grad_hidden = grad_combined[:, self.input_size:]

        # set accumulated gradients
        self.fc_hidden.gradient_weights = accumulated_grad_hidden
        self.fc_output.gradient_weights = accumulated_grad_output

        # update weights if optimizer is set
        if self._optimizer is not None:
            self.fc_hidden.weights = self._optimizer.calculate_update(
                self.fc_hidden.weights, self.fc_hidden.gradient_weights
            )
            # need separate optimizer for output layer
            if not hasattr(self, '_optimizer_output'):
                self._optimizer_output = copy.deepcopy(self._optimizer)
            self.fc_output.weights = self._optimizer_output.calculate_update(
                self.fc_output.weights, self.fc_output.gradient_weights
            )

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        loss = 0
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            loss += self._optimizer.regularizer.calculate_norm(self.fc_hidden.weights)
            loss += self._optimizer.regularizer.calculate_norm(self.fc_output.weights)
        return loss

