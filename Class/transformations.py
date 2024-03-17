
import numpy as np
from module import Module

class TanH(Module):
    def __init__(self):
        super().__init__()
        self.name = 'tanh'

    def forward(self, data):
        return np.tanh(data)

    def backward_delta(self, data, delta):
        return delta * (1 - self(data)**2)
    
    def zero_grad(self):
        pass
    def update_parameters(self, gradient_step=1e-3):
        pass
    def backward_update_gradient(self, data, delta):
        pass
    def predict(self, X):
        pass

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'

    def forward(self, data):
        return 1 / (1 + np.exp(-data))

    def backward_delta(self, data, delta):
        return delta * (1 - self(data)) * self(data)
    
    def zero_grad(self):
        pass
    def update_parameters(self, gradient_step=1e-3):
        pass
    def backward_update_gradient(self, data, delta):
        pass
    def predict(self, X):
        pass