import numpy as np
from module import Loss

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
    
