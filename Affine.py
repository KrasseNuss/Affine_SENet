import numpy as np

class AffineToLinear():
    @staticmethod
    #notwendige Transformation, um affine Subspaces in lineare Subspaces im erweiterten Raum zu überführen
    #Für SSC
    def makeLinear(x: np.ndarray) -> np.ndarray:
        ones = np.ones((1, x.shape[1]))
        x_linear = np.vstack((x, ones))
        return x_linear
    
    #Für SENet
    @staticmethod
    def makeLinearSENet(x: np.ndarray) -> np.ndarray:
        ones = np.ones((x.shape[1], 1))
        x_linear = np.vstack((x, ones))
        return x_linear
    
    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        x_norm = x / np.linalg.norm(x, axis=0, keepdims=True)
        return x_norm

