import numpy as np
from module import Loss


class MSELoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: np.array, yhat: np.array):
        """
        :param y: np.array shape : (batch_size , d)
        :param yhat: np.array shape : (batch_size , d)
        :return: np.array shape : (batch_size ,)
        """
        assert y.shape == yhat.shape, ValueError(
            f"y et yhat doivent avoir la meme dimension."
        )

        return np.sum((y - yhat) ** 2, axis=1)

    def backward(self, y, yhat):
        """
        :param y: batch x d
        :param yhat: batch x d
        :return: batch x d
        """

        assert y.shape == yhat.shape, ValueError(
            f"y et yhat doivent avoir la meme dimension."
        )

        return 2 * (yhat - y)


class CrossEntropyLoss(Loss):
    def forward(self, y: np.array, yhat: np.array):
        """
        :param y: np.array shape : (batch_size , 1) (entiers entre 0 et d-1)
        :param yhat: np.array shape : (batch_size , d) (nombres entre 0 et 1)
        :return: np.array shape : (batch_size ,)
        """
        y = y.flatten()

        return -np.log(yhat[np.arange(y.size), y])

    def backward(self, y: np.array, yhat: np.array):
        """

        :param y: np.array shape : (batch_size , 1) (entiers entre 0 et d-1)
        :param yhat: np.array shape : (batch_size , d) (nombres entre 0 et 1)
        :return: np.array shape : (batch_size , d)
        """
        y = y.flatten()
        onehot = np.zeros_like(yhat)
        onehot[np.arange(y.size), y] = 1

        return -onehot / (yhat + 1e-12)


class LogSoftMaxCrossEntropy(Loss):
    def forward(self, y: np.array, yhat: np.array):
        """
        :param y: np.array shape : (batch_size , 1) (entiers entre 0 et d-1)
        :param yhat: np.array shape : (batch_size , d) (nombres entre 0 et 1)
        :return: np.array shape : (batch_size ,)
        """
        y = y.flatten()
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        return (
            -yhat[np.arange(y.size), y]
            + np.log(np.sum(np.exp(yhat - yhat_max), axis=1))
            + yhat_max.flatten()
        )

    def backward(self, y: np.array, yhat: np.array):
        """
        :param y: np.array shape : (batch_size , 1) (entiers entre 0 et d-1)
        :param yhat: np.array shape : (batch_size , d) (nombres entre 0 et 1)
        :return: np.array shape : (batch_size , d)
        """
        y = y.flatten()
        onehot = np.zeros_like(yhat)
        onehot[np.arange(y.size), y] = 1
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        expy = np.exp(yhat - yhat_max)
        return yhat - onehot + expy / np.sum(expy, axis=1).reshape(-1, 1)
