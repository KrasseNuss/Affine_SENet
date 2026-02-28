"""
Generate data with subspace clusters depending on users input parameters
Parameters used:
    -      n : number of points in the dataset
    -  sub_n : number of points in one subspace cluster area
    -      d : number of dimensions overall
    -  sub_d : number of dimensions used for a subspace cluster
    -      c : number of cluster centers per subspace cluster blob (default: 1)
    -    std : standard derivation of the cluster(s) (default: 1.0)
    - mu_clu : Boolean if it is possible for one point to be in multiple subspaces
    TODO Tune hyperparameters in random subspaces
"""

# INFO ----------------------------------------------------------------------------------------------------------------
# Author: Nanni Schueler
# Version: 0.3
# Date: 2019-06-11
# ---------------------------------------------------------------------------------------------------------------------

# IMPORT --------------------------------------------------------------------------------------------------------------
import numpy as np
import math
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# PARAMS --------------------------------------------------------------------------------------------------------------
#subspace_cluster
#subspace_lables
#m_pointer
#empty_space
#number_sub_clusters


# ---------------------------------------------------------------------------------------------------------------------
def print_instructions():
    print("This function needs the following parameters:")
    print(" -                     n (int): The number of points you want to have in your dataset")
    print(" -                     d (int): The number of dimensions per point")
    print(" -               mu_clu (bool): Can one point be included in multiple subspaces? [Default:False]")
    print("[Optional] subspaces (ndarray): A list of subspaces you want to have in your dataset, written "
          "in the following way:")
    print("                                [[sub_n, sub_d, c, std], ... ] with")
    print("                                  - sub_n   (int): number of points in this subspace")
    print("                                  - sub_d   (int): number of dimensions in this subspace")
    print("                                  -     c   (int): number of clusters in this subspace")
    print("                                  -   std (float): the standard derivation of the cluster(s)")
    print(" Example:")
    print(" A, lables = generate_subspacedata(10, 10, True, [[2, 4, 1, 1.0], [3, 2, 2, 1.0], [6, 3, 1, 0.4], "
          "[4, 9, 1, 0.6]])")


def generate_subspacedata(n=0, d=0, mu_clu=False, subspaces=None):
    if n == 0 or d == 0:
        print_instructions()
        return None, None
    elif subspaces is None:
        #print("I will generate random subspaces...")
        return random_subspaces(n, d, mu_clu)
    else:
        subspace_cluster = np.zeros((n, d))
        subspace_lables = np.zeros((n, d))
        empty_space = []
        empty_rows = 0
        n_pointer = 0
        needed_dim = -1
        number_sub_clusters = 0
        for sub in subspaces:
            # print(str(sub) + " und der ganze Rest: "+str(d_space)+" und "+str(n_pointer))
            number_sub_clusters = number_sub_clusters + 1
            sub_n = sub[0]
            sub_d = sub[1]
            c = sub[2]
            std = sub[3]
            x, y = make_subspaceblob(sub_n, sub_d, c, std)
            if sub_d <= d:
                if sub_n <= (n - n_pointer):
                    for i in range(0, len(x)):
                        for j in range(0, len(x[0])):
                            subspace_cluster[i+n_pointer][j] = x[i][j]
                            subspace_lables[i+n_pointer][j] = number_sub_clusters
                        empty_space.append(sub_d)
                    n_pointer = n_pointer + sub_n
                else:
                    # place behind other clusters if enabled
                    if mu_clu:
                        while len(empty_space) < d:
                            empty_space.append(0)
                        #print(empty_space)
                        for l in empty_space:
                            if d - l >= sub_d:
                                empty_rows = empty_rows + 1
                                needed_dim = max(l, needed_dim)
                                if empty_rows >= sub_n:
                                    for i in range(0, len(x)):
                                        for j in range(0, len(x[0])):
                                            subspace_cluster[i][j + needed_dim] = x[i][j]
                                            subspace_lables[i][j + needed_dim] = number_sub_clusters
                        #print(needed_dim)
                    else:
                        print("There are more points in the subspaces than in your dataset!\n"
                              "Either change those numbers or enable points to be in multiple subspaces by adding "
                              "the parameter 'True'.")
            else:
                print("The subspace cluster has more dimensions than your original dataspace.")

        for i in range(0, len(subspace_cluster)):
            for j in range(0, len(subspace_cluster[0])):
                if subspace_cluster[i][j] == 0:
                    subspace_cluster[i][j]= np.random.uniform(-100, 100)
        
        np.savetxt("subspace_cluster.csv", subspace_cluster, delimiter=",")
        np.savetxt("subspace_lables.csv", subspace_lables, delimiter=",")
        return subspace_cluster, subspace_lables


def make_subspaceblob(sub_n, sub_d, c, std):
    box = 100
    #print("Making "+str(c)+" subspace cluster(s) with "+str(sub_d)+" dimensions over "+str(sub_n)+" points (std: "+str(std)+").")
    X, y = make_blobs(n_samples=sub_n, n_features=sub_d, centers=c, cluster_std=std, center_box=(-box, box), shuffle=True, random_state=None)
    return X, y


def random_subspaces(n, d, mu_clu):
    subspaces = []

    max_clusters = max(1, int(np.sqrt(np.sqrt(n * d))))
    number_random_clusters = random.randint(1, max_clusters)

    for _ in range(number_random_clusters):

        sub_n_min = max(1, int(0.05 * n))
        sub_n_max = max(sub_n_min, int(0.8 * n / number_random_clusters))

        sub_d_min = max(1, int(0.1 * d))
        sub_d_max = max(sub_d_min, int(0.6 * d))

        sub_n = random.randint(sub_n_min, sub_n_max)
        sub_d = random.randint(sub_d_min, sub_d_max)

        c = random.randint(1, max(1, sub_d // 2))
        std = round(random.uniform(0.1, 1.0), 2)

        subspaces.append([sub_n, sub_d, c, std])

    print("Subspaces are:", subspaces)
    return generate_subspacedata(n, d, mu_clu, subspaces)

#Automatisches Cluster generieren funktioniert nicht, da das nur noch Rauschen ist.
# TEST ----------------------------------------------------------------------------------------------------------------
#A, lables = generate_subspacedata(200, 10, False, [[2, 4, 1, 1.0], [3, 6, 2, 1.0], [1, 5, 1, 0.4], [4, 9, 1, 0.6]])
A, lables = generate_subspacedata(200, 2, False, [[100, 1, 1, 1.0], [80, 1, 1, 0.9]])
#A, lables = generate_subspacedata()
plt.plot(A[:, 0], A[:, 1], 'o')
plt.show()
print(A)
print(lables)
