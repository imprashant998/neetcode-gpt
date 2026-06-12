import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        # -----------------------
        # assuming the y_pred are the predicted probabilities of the label being '1'
        epsilon = 10**-7
        y_pred = y_pred + epsilon
        class_1_loss = -1* np.dot(y_true, np.log(y_pred))
        class_2_loss = -1* np.dot(1-y_true, np.log(1-y_pred))
        result = class_1_loss + class_2_loss
        result = result / len(y_true)
        return np.round(result,4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        N = y_true.shape[0]
        C = y_true.shape[1]
        epsilon = 10**-7
        y_pred = y_pred + epsilon
        loss = 0
        for i in range(N):
            y_pred_i = y_pred[i]
            y_true_i = y_true[i]
            loss += -1* np.dot(y_true_i, np.log(y_pred_i))
        loss = loss / N
        return np.round(loss,4)


