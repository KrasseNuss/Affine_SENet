import numpy as np

class AffineToLinear():
    @staticmethod
    #notwendige Transformation, um affine Subspaces in lineare Subspaces im erweiterten Raum zu überführen
    def makeLinear(x: np.ndarray) -> np.ndarray:
        ones = np.ones((x.shape[0], 1))
        x_linear = np.hstack((x, ones))
        return x_linear
    
