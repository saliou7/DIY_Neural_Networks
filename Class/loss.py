import numpy as np
from module import Loss

class MSELoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y:np.array, yhat:np.array):
        """
            :param y: np.array shape : (batch_size , d)
            :param yhat: np.array shape : (batch_size , d)
            :return: np.array shape : (batch_size ,)
        """
        assert y.shape == yhat.shape, ValueError(f"y et yhat doivent avoir la meme dimension.")

        return np.sum((y - yhat) ** 2, axis=1)

    def backward(self, y, yhat):
        """
            :param y: batch x d
            :param yhat: batch x d
            :return: batch x d
        """

        assert y.shape == yhat.shape, ValueError(f"y et yhat doivent avoir la meme dimension.")

        return 2 * (yhat - y)
    
