import numpy as np


class Constant:
    """Init weights with a constant value."""

    def __init__(self, constant: float = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in: int, fan_out: int):
        """Returns an initialized tensor witht the desired shape"""
        weights = np.ones(weights_shape) * self.constant

        return weights


class UniformRandom:
    """Init weights randomly with normal distributed values."""

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in: int, fan_out: int):
        """Returns an initialized tensor witht the desired shape"""
        weights = np.random.uniform(0, 1, size=weights_shape)

        return weights


class Xavier:
    """Use Xavier/Glorot Initializiation."""

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in: int, fan_out: int):
        """Returns an initialized tensor witht the desired shape"""
        std = np.sqrt(2 / (fan_out + fan_in))  # std of xavier init
        weights = np.random.normal(0, std, size=weights_shape)

        return weights


class He:
    """Use He Initializiation."""

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in: int, fan_out: int):
        """Returns an initialized tensor witht the desired shape"""

        std = np.sqrt(2 / fan_in)  # std of He init
        weights = np.random.normal(0, std, size=weights_shape)

        return weights
