import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

    def predict(self, X):
        ## Calcule la prediction
        pass


class MSELoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y, yhat):
        """
        calcule l'erreur quadratique moyenne entre y et yhat
        """
        assert y.shape == yhat.shape, ValueError(f"y et yhat doivent avoir la meme dimension.")
        self._y = y
        self._yhat = yhat
        return np.mean((y - yhat) ** 2)

    def backward(self, y, yhat):
        """
        calcule la derivee de l'erreur par rapport a yhat
        """
        assert y.shape == yhat.shape, ValueError(f"y et yhat doivent avoir la meme dimension.")
        return -2 * (y - yhat)
    
    
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        #Initialize weights en utilisant une distribution normale
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        # Initialize bias comme un vecteur de zeros
        self.bias = np.zeros((1, output_dim))
        self._grad_weights = np.zeros_like(self.weights)
        self._grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        # Ensure input is two-dimensional (batch_size, input_dim)
        assert x.shape[1] == self.input_dim, "Input x must have shape (batch_size, input_dim)"
        self._x = x  # Store for use in backward pass
        return np.dot(x, self.weights) + self.bias

    def backward_update_gradient(self, input, delta):
        # Ensure delta is two-dimensional and matches the output shape
        assert input.shape[1] == self.input_dim, "Input must have shape (batch_size, input_dim)"
        assert delta.shape[1] == self.output_dim, "Delta must have shape (batch_size, output_dim)"

        self._grad_weights = np.dot(input.T, delta)
        self._grad_bias = np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self.input_dim, "Input must have shape (batch_size, input_dim)"
        assert delta.shape[1] == self.output_dim, "Delta must have shape (batch_size, output_dim)"

        # Calculate the derivative of the error with respect to the input
        return np.dot(delta, self.weights.T) 

    def zero_grad(self):
        self._grad_weights.fill(0)
        self._grad_bias.fill(0)

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self._grad_weights
        self.bias -= learning_rate * self._grad_bias

    def predict(self, x):
        return np.where(self.forward(x) >= 0.5, 1, 0)
