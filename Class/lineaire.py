import numpy as np
from module import Module

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()  # Call the constructor of the parent class
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        #Initialize weights en utilisant une distribution normale
        self._parameters["weight"] = np.random.randn(input_dim, output_dim) * 0.01

        # Initialize bias comme un vecteur de zeros
        self._parameters["bias"] = np.zeros((1, output_dim))

        # Initialize gradient
        self._gradient["weight"] = np.zeros_like(self._parameters["weight"])

        # Initialize bias gradient
        self._gradient["bias"] = np.zeros_like(self._parameters["bias"])

    def forward(self, x):
        assert x.shape[1] == self.input_dim, "Input x must have shape (batch_size, input_dim)"
        # self._x = x  # Store for use in backward pass  (not needed and i don't know why you use it 😏)
        return np.dot(x, self._parameters["weight"]) + self._parameters["bias"]

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self.input_dim, "Input must have shape (batch_size, input_dim)"
        assert delta.shape[1] == self.output_dim, "Delta must have shape (batch_size, output_dim)"

        self._gradient["weight"] += np.dot(input.T, delta)
        self._gradient["bias"] += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self.input_dim, "Input must have shape (batch_size, input_dim)"
        assert delta.shape[1] == self.output_dim, "Delta must have shape (batch_size, output_dim)"

        # Calculate the derivative of the error with respect to the input
        return np.dot(delta, self._parameters["weight"].T)

    def zero_grad(self):
        self._gradient["weight"].fill(0)
        self._gradient["bias"].fill(0)

    def update_parameters(self, learning_rate):
        self._parameters["weight"] -= learning_rate * self._gradient["weight"]
        self._parameters["bias"] -= learning_rate * self._gradient["bias"]

    def predict(self, x):
        return np.where(self.forward(x) >= 0.5, 1, 0)