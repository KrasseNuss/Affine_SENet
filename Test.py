import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np

#Import aus Repo
from subspace_clustering.cluster.selfrepresentation import ElasticNetSubspaceClustering

# generate 7 data points from 3 independent subspaces as columns of data matrix X
X = np.array([[1.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0],
              [1.0,  0.5, 0.0, 0.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 1.0, 0.2, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.2, 1.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0,  1.0, -1.0]])

model = ElasticNetSubspaceClustering(n_clusters=3,algorithm='lasso_lars',gamma=50).fit(X.T)
print(model.labels_)
# this should give you array([1, 1, 0, 0, 2, 2, 2]) or a permutation of these labels