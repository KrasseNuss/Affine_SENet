import sys
import os
sys.path.append(os.path.abspath("."))
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from self_expressive_network.metrics.cluster.accuracy import clustering_accuracy
import numpy as np
from Affine import AffineToLinear as alt
import time
import random

#Import aus Repo
from subspace_clustering.cluster.selfrepresentation import ElasticNetSubspaceClustering

def random_gamma():
    gamma = np.random.uniform(0.1, 100)
    return gamma

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Seeds set to: {seed}")

def run_experiment(gamma, affine_model, cluster, labels,test):
    if test:
        print("Model: ")
        print(affine_model)
        print("\n")
    model_affine = ElasticNetSubspaceClustering(n_clusters=2,algorithm='lasso_lars',gamma=gamma).fit(affine_model)
    if test:
        print("Labels: ")
        print(labels)
        print("Labels: ")
        print(model_affine.labels_)
        print("-------------------------------------------")
        print("cluster shape:", cluster.shape)
        print("Labels Shape: ", labels.shape)
        print("affine_model shape:", affine_model.shape)
        print("predicted labels length:", len(model_affine.labels_))
        print("affine Labels length:", model_affine.labels_.shape)
        print("true labels length:", len(labels))
        print("-------------------------------------------")

    ari = adjusted_rand_score(labels, model_affine.labels_)
    nmi = normalized_mutual_info_score(labels, model_affine.labels_, average_method='arithmetic')
    acc = clustering_accuracy(labels, model_affine.labels_)
    return ari, nmi, acc

def main():
    test = False
    start = time.perf_counter()
    cluster = np.loadtxt("/mnt/d/Xaver Köppl/Uni/Bachelorarbeit/git/SubCluGen/subspace_cluster.csv", delimiter=",", dtype=np.float64)
    labels = np.loadtxt("/mnt/d/Xaver Köppl/Uni/Bachelorarbeit/git/SubCluGen/subspace_lables.csv", delimiter=",", dtype=np.float64)
    labels = np.max(labels, axis=1)
    #print("Data: ")
    #print(cluster)
    affine_model = alt.makeLinear(cluster)

    trial_num = 50
    best_score = -1
    best_gamma = None
    for trial in range(trial_num):
        gamma = random_gamma()
        same_seeds(int(gamma) + trial)
        print(f"Trial {trial} with gamma={gamma:.4f}")

        ari, nmi, acc = run_experiment(gamma, affine_model, cluster, labels, test)

        if ari > best_score:
            best_score = ari
            best_gamma = gamma

    trial_time = time.perf_counter() - start
    print(f"Best gamma: {best_gamma}")
    print(f"Best ARI: {best_score}")
    print(f"Trial time: {trial_time:.2f} seconds")

    final_run = 5
    results = []

    for i in range(final_run):
        same_seeds(int(best_gamma) + i)
        ari, nmi, acc = run_experiment(best_gamma, affine_model, cluster, labels, test)
        results.append((acc, nmi, ari))

    results = np.array(results)

    endtime = time.perf_counter() - start
    print("ARI: {:.4f} ± {:.4f}".format(results[:, 0].mean(), results[:, 0].std()))
    print("NMI: {:.4f} ± {:.4f}".format(results[:, 1].mean(), results[:, 1].std()))
    print("ACC: {:.4f} ± {:.4f}".format(results[:, 2].mean(), results[:, 2].std()))
    print(f"Best gamma: {best_gamma}")
    print("Total time: {:.2f} seconds".format(endtime))
    print("\n")
    return results, endtime