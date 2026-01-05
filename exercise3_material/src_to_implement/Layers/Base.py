class BaseLayer:
    def __init__(self, trainable: bool = False, testing_phase: bool = False):
        self.trainable = trainable
        self.testing_phase = testing_phase
