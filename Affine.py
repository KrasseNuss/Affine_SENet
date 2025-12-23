import numpy as np

class AffineToLinear():
    @staticmethod
    def makeLinear(x: np.ndarray):
        ones = np.ones((1, x.shape[1]))
        x_linear = np.vstack((x, ones))
        return x_linear
    
    def normalize(x: np.ndarray):
        x_norm = x / np.linalg.norm(x, axis=0, keepdims=True)
        return x_norm

