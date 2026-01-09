import numpy as np
from Layers import Base


class Sigmoid(Base.BaseLayer):
    """
    Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))

    Maps any real value to (0, 1), useful for binary classification
    or gates in recurrent networks (LSTM, GRU).
    """

    def __init__(self):
        super().__init__(trainable=False)
        self.activation = None

    def forward(self, input_tensor):
        # σ(x) = 1 / (1 + exp(-x))
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        # Derivative: σ'(x) = σ(x) * (1 - σ(x))
        # Chain rule: dL/dx = dL/dσ * σ * (1 - σ)
        return error_tensor * self.activation * (1 - self.activation)
