import numpy as np
from module import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.inputs = []

    def forward(self, X):
        self.inputs = [X]
        for module in self.modules:
            X = module.forward(X)
            self.inputs.append(X)
        return X

    def backward_update_gradient(self, input, delta):
        for i, module in enumerate(reversed(self.modules)):
            module.backward_update_gradient(self.inputs[-i-2], delta)
            delta = module.backward_delta(self.inputs[-i-2], delta)

    def backward_delta(self, input, delta):
        for module in reversed(self.modules):
            delta = module.backward_delta(module.forward(input), delta)
            input = module.forward(input)
        return delta

    def update_parameters(self, learning_rate=1e-3):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

class Optim(object):
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        # Forward pass
        output = self.net.forward(batch_x)
        
        # Compute loss
        loss_value = np.mean(self.loss.forward(batch_y, output))
        
        # Backward pass
        gradient = self.loss.backward(batch_y, output)
        self.net.zero_grad()
        self.net.backward_update_gradient(batch_x, gradient)
        self.net.update_parameters(self.eps)
        
        return loss_value

    def sgd(self, x_train: np.ndarray, y_train: np.ndarray, batch_size: None, epochs: int):
        num_samples = x_train.shape[0]
        for epoch in range(epochs):
            if batch_size == None:
                batch_size = num_samples
            for i in range(0, num_samples, batch_size):
                batch_x = x_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                self.step(batch_x, batch_y)
