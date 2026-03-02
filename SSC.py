import sys
import os
sys.path.append(os.path.abspath("."))
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from self_expressive_network.metrics.cluster.accuracy import clustering_accuracy
import numpy as np
from Affine import AffineToLinear as alt

#Import aus Repo
from subspace_clustering.cluster.selfrepresentation import ElasticNetSubspaceClustering

test = True
# generate 7 data points from 3 independent subspaces as columns of data matrix X
#X = np.array([[1.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0],
#              [1.0,  0.5, 0.0, 0.0, 0.0,  0.0, 0.0],
#              [0.0,  0.0, 1.0, 0.2, 0.0,  0.0, 0.0],
#              [0.0,  0.0, 0.2, 1.0, 0.0,  0.0, 0.0],
#              [0.0,  0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
#              [0.0,  0.0, 0.0, 0.0, 1.0,  1.0, -1.0]])

cluster = np.loadtxt("/mnt/d/Xaver Köppl/Uni/Bachelorarbeit/git/SubCluGen/subspace_cluster.csv", delimiter=",", dtype=np.float64)
labels = np.loadtxt("/mnt/d/Xaver Köppl/Uni/Bachelorarbeit/git/SubCluGen/subspace_lables.csv", delimiter=",", dtype=np.float64)
labels = np.max(labels, axis=1)
#model = ElasticNetSubspaceClustering(n_clusters=1,algorithm='lasso_lars',gamma=50).fit(A.T)
#print(model.labels_)
# this should give you array([1, 1, 0, 0, 2, 2, 2]) or a permutation of these labels
#print("Data: ")
#print(cluster)
#print("\n")
affine_model = alt.makeLinear(cluster)
print("Model: ")
print(affine_model)
print("\n")
model_affine = ElasticNetSubspaceClustering(n_clusters=2,algorithm='lasso_lars',gamma=10).fit(affine_model)
print("Labels: ")
print(labels)
print("Labels: ")
print(model_affine.labels_)
if test:
    print("-------------------------------------------")
    print("cluster shape:", cluster.shape)
    print("Labels Shape: ", labels.shape)
    print("affine_model shape:", affine_model.shape)
    print("predicted labels length:", len(model_affine.labels_))
    print("affine Labels length:", model_affine.labels_.shape)
    print("true labels length:", len(labels))
    print("-------------------------------------------")
#labels = np.max(labels, axis=1)
ari = adjusted_rand_score(labels, model_affine.labels_)
nmi = normalized_mutual_info_score(labels, model_affine.labels_, average_method='arithmetic')
acc = clustering_accuracy(labels, model_affine.labels_)
print("ARI: {:.4f}".format(ari))
print("NMI: {:.4f}".format(nmi))
print("ACC: {:.4f}".format(acc))
print("\n")