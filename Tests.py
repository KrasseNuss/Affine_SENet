import sys
import os
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./self_expressive_network"))
from SSC import main as ssc_main
from self_expressive_network.main import main as sen_main
from SubCluGen.generator import generate_subspacedata
import numpy as np
import time

#Variavles for testing
N = 300 #number of points
N_values = [300, 600, 900, 1200, 1500] #array with number of points
D = 2 #number of dimensions
MU_CLU = False #option if a point is memeber of multiple clusters
SUBSPACES = [[150, 1, 1, 1.0], [150, 1, 1, 0.9]] #list of subspaceswith the following format: [number of points, number of dimensions, number of clusters, derivation of the cluster]
num_trials = 5

def change_N(N): 
    N = N*2
    return N

def change_D(D):
    D = D+1
    return D

def change_MU_CLU(MU_CLU):
    MU_CLU = not MU_CLU
    return MU_CLU

def scale_SUBSPACES(SUBSPACES, N_current):
    new_subspaces = []
    points = N_current // 2
    for sub in SUBSPACES:
        new_subspaces.append([points, sub[1], sub[2], sub[3]])
    return new_subspaces

def new_points_SUBSPACES(num):
    global N, D
    points = int(N // num)
    new_subspaces = []
    for i in range(num):
        new_subspaces.append([points, 1, 1, 0.9])
    return new_subspaces

def new_dim_SUBSPACES(num):
    global N, D
    points = int(N // num)
    new_subspaces = []
    for i in range(num):
        new_subspaces.append([points, num, 1, 0.9])
    return new_subspaces

def values(results):
    if not results:
        print("Warning: No results for this experiment.")
        return (0, 0, 0, 0, 0, 0, 0, 0)

    results = np.array(results)
    mean_ari = results[:, 0].mean()
    std_ari = results[:, 0].std()
    mean_nmi = results[:, 1].mean()
    std_nmi = results[:, 1].std()
    mean_acc = results[:, 2].mean()
    std_acc = results[:, 2].std()
    mean_time = results[:,3].mean()
    std_time = results[:,3].std()
    return (mean_ari, std_ari, mean_nmi, std_nmi, mean_acc, std_acc, mean_time, std_time)

def run_experiment_N():
    print("Testing with changing N:")
    N_current = N_values[0]
    current_subspaces = SUBSPACES
    ssc_results_final = []
    senet_results_final = []
    ssc_results = []
    senet_results = []
    sub_list = []
    generate_subspacedata(int(N_current), int(D), bool(MU_CLU), current_subspaces)
    time.sleep(1)
    for i in range(5):
        try:
            results_ssc, time_ssc = ssc_main()
            results_senet, time_senet = sen_main()
        except Exception as e:
            print(f"Error in run {i} with the Parameters N={N_current}, D={D}, Subspaces={current_subspaces}: {e}")
            with open("error_log_N.txt", "a") as log:
                log.write(f"Error in run {i} with the Parameters N={N_current}, D={D}, Subspaces={current_subspaces}: {str(e)}\n")
            continue
        ssc_results.append([results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
        senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])
    mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
    mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
    ssc_results_final.append((N_current, D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
    senet_results_final.append((N_current, D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
    sub_list.append(current_subspaces)
    for j in range(2, num_trials):
        N_current = N_values[j-1]
        current_subspaces = scale_SUBSPACES(current_subspaces, N_current)
        generate_subspacedata(int(N_current), int(D), bool(MU_CLU), current_subspaces)
        time.sleep(1)
        ssc_results = []
        senet_results = []
        for z in range(5):
            try:
                results_ssc, time_ssc = ssc_main()
                results_senet, time_senet = sen_main()
            except Exception as e:
                print(f"Error in run {z} with the Parameters N={N_current}, D={D}, Subspaces={current_subspaces}: {e}")
                with open("error_log_N.txt", "a") as log:
                    log.write(f"Error in run {z} with the Parameters N={N_current}, D={D}, Subspaces={current_subspaces}: {str(e)}\n")
                continue
            ssc_results.append([results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
            senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])
        mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
        mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
        ssc_results_final.append((N_current, D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
        senet_results_final.append((N_current, D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
        sub_list.append(current_subspaces)
    np.savetxt("ssc_results_N.csv", ssc_results_final, delimiter=",")
    np.savetxt("senet_results_N.csv", senet_results_final, delimiter=",")
    np.savetxt("current_sub_N.txt", sub_list, delimiter=",")

def run_experiment_D():
    print("Testing with changing D:")
    current_D = D
    ssc_results_final = []
    senet_results_final = []
    ssc_results = []
    senet_results = []
    generate_subspacedata(int(N), int(current_D), bool(MU_CLU), SUBSPACES)
    time.sleep(1)
    for i in range(5):
        try:
            results_ssc, time_ssc = ssc_main()
            results_senet, time_senet = sen_main()
        except Exception as e:
            print(f"Error in run {i} with the Parameters N={N}, D={current_D}, Subspaces={SUBSPACES}: {e}")
            with open("error_log_D.txt", "a") as log:
                log.write(f"Error in run {i} with the Parameters N={N}, D={current_D}, Subspaces={SUBSPACES}: {str(e)}\n")
            continue
        ssc_results.append([results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
        senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])
    mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
    mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
    ssc_results_final.append((N, current_D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
    senet_results_final.append((N, current_D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
    for j in range(2, num_trials):
        current_D = change_D(current_D)
        generate_subspacedata(int(N), int(current_D), bool(MU_CLU), SUBSPACES)
        time.sleep(1)
        ssc_results = []
        senet_results = []
        for z in range(5):
            try:
                results_ssc, time_ssc = ssc_main()
                results_senet, time_senet = sen_main()
            except Exception as e:
                print(f"Error in run {z} with the Parameters N={N}, D={current_D}, Subspaces={SUBSPACES}: {e}")
                with open("error_log_D.txt", "a") as log:
                    log.write(f"Error in run {z} with the Parameters N={N}, D={current_D}, Subspaces={SUBSPACES}: {str(e)}\n")
                continue
            ssc_results.append([results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
            senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])
        mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
        mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
        ssc_results_final.append((N, current_D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
        senet_results_final.append((N, current_D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
    np.savetxt("ssc_results_D.csv", ssc_results_final, delimiter=",")
    np.savetxt("senet_results_D.csv", senet_results_final, delimiter=",")
    np.savetxt("subspaces_D.txt", SUBSPACES, delimiter=",")

def run_experiment_SUB():
    print("Testing with changing the number of subspaces:")
    current_sub = SUBSPACES
    ssc_results = []
    senet_results = []
    ssc_results_final = []
    senet_results_final = []
    sub_list = []
    generate_subspacedata(int(N), int(D), bool(MU_CLU), current_sub)
    time.sleep(1)
    for i in range(5):
        try:
            results_ssc, time_ssc = ssc_main()
            results_senet, time_senet = sen_main()
        except Exception as e:
            print(f"Error in run {i} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {e}")
            with open("error_log_SUB.txt", "a") as log:
                log.write(f"Error in run {i} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {str(e)}\n")
            continue
        ssc_results.append([ results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
        senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])    
    mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
    mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
    ssc_results_final.append((N, D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
    senet_results_final.append((N, D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
    sub_list.append(current_sub)
    for j in range(2, num_trials):
        current_sub = new_points_SUBSPACES(j)
        generate_subspacedata(int(N), int(D), bool(MU_CLU), current_sub)
        time.sleep(1)
        ssc_results = []
        senet_results = []
        for z in range(5):
            try:
                results_ssc, time_ssc = ssc_main()
                results_senet, time_senet = sen_main()
            except Exception as e:
                print(f"Error in run {z} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {e}")
                with open("error_log_SUB.txt", "a") as log:
                    log.write(f"Error in run {z} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {str(e)}\n")
                continue
            ssc_results.append([results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
            senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])        
        mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
        mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
        ssc_results_final.append((N, D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
        senet_results_final.append((N, D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
        sub_list.append(current_sub)
    np.savetxt("ssc_results_SUBSPACES.csv", ssc_results_final, delimiter=",")
    np.savetxt("senet_results_SUBSPACES.csv", senet_results_final, delimiter=",")
    np.savetxt("current_sub_SUBSPACES.txt", sub_list, delimiter=",")

def run_experiment_SUB_DIM():
    print("Testing with changing the dimensions of subspaces:")
    current_sub = SUBSPACES
    ssc_results = []
    senet_results = []
    ssc_results_final = []
    senet_results_final = []
    sub_list = []
    generate_subspacedata(int(N), int(D), bool(MU_CLU), current_sub)
    time.sleep(1)
    for i in range(5):
        try:
            results_ssc, time_ssc = ssc_main()
            results_senet, time_senet = sen_main()
        except Exception as e:
            print(f"Error in run {i} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {e}")
            with open("error_log_SUB_DIM.txt", "a") as log:
                log.write(f"Error in run {i} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {str(e)}\n")
            continue
        ssc_results.append([ results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
        senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])    
    mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
    mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
    ssc_results_final.append((N, D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
    senet_results_final.append((N, D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
    sub_list.append(current_sub)
    for j in range(2, num_trials):
        current_sub = new_dim_SUBSPACES(j)
        generate_subspacedata(int(N), int(D), bool(MU_CLU), current_sub)
        time.sleep(1)
        ssc_results = []
        senet_results = []
        for z in range(5):
            try:
                results_ssc, time_ssc = ssc_main()
                results_senet, time_senet = sen_main()
            except Exception as e:
                print(f"Error in run {z} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {e}")
                with open("error_log_SUB_DIM.txt", "a") as log:
                    log.write(f"Error in run {z} with the Parameters N={N}, D={D}, Subspaces={current_sub}: {str(e)}\n")
                continue
            ssc_results.append([results_ssc[0], results_ssc[1], results_ssc[2], time_ssc])
            senet_results.append([results_senet[0], results_senet[1], results_senet[2], time_senet])        
        mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc = values(ssc_results)
        mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet = values(senet_results)
        ssc_results_final.append((N, D, mean_ari_ssc, std_ari_ssc, mean_nmi_ssc, std_nmi_ssc, mean_acc_ssc, std_acc_ssc, mean_time_ssc, std_time_ssc))
        senet_results_final.append((N, D, mean_ari_senet, std_ari_senet, mean_nmi_senet, std_nmi_senet, mean_acc_senet, std_acc_senet, mean_time_senet, std_time_senet))
        sub_list.append(current_sub)
    np.savetxt("ssc_results_SUBSPACES_DIM.csv", ssc_results_final, delimiter=",")
    np.savetxt("senet_results_SUBSPACES_DIM.csv", senet_results_final, delimiter=",")
    np.savetxt("current_sub_SUBSPACES_DIM.txt", sub_list, delimiter=",")

if __name__ == "__main__":
    #Changing of N
    run_experiment_N()

    #Changing of D
    run_experiment_D()
        
    #Changing number of points in subspaces
    run_experiment_SUB()

    #Changing dimensions of subspaces
    run_experiment_SUB_DIM()