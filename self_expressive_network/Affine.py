import numpy as np

class AffineToLinear():
    @staticmethod
    def makeLinear(x: np.ndarray) -> np.ndarray:
        ones = np.ones((x.shape[0], 1))
        x_linear = np.hstack((x, ones))
        return x_linear
    
