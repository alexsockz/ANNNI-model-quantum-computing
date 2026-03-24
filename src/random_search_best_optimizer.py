import os
import warnings
from VQE import VQE
import torch
from time import perf_counter
import numpy as np
import optuna
import optuna.visualization as vis
from tqdm import tqdm

from jax import numpy as jnp
from jax import vmap
import pennylane as qml
from random import randint
# Suppress JAX and numerical warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ["JAX_QUIET_STARTUP"] = "1"
# Disable CUDA entirely to avoid CuDNN version mismatch error
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

torch.set_num_threads(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_H(num_spins, k, h):
    """Construction function for the ANNNI Hamiltonian (J=1)"""
    # Interaction between spins (neighbouring):
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H - (qml.PauliX(i) @ qml.PauliX(i + 1))

    # Interaction between spins (next-neighbouring):
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))

    # Interaction of the spins with the magnetic field
    for i in range(0, num_spins):
        H = H - h * qml.PauliZ(i)

    return H

def run_trial(trial):
    #paper says 6/12
    #n_qubits = trial.suggest_categorical("n_qubits", [4, 6, 8, 12])
   
    ansatz_depth = trial.suggest_int("ansatz_depth", 5, 9)
    
    #n_shots = trial.suggest_categorical("n_shots", [100, 1000, 10000])
    #SETTING IT TO NONE FOR TIME IT WOULD BE MORE CORRECT TO HAVE IT WITH SET NUMBER OF SHOTS
    n_shots=1000
    
    # Suggesting from a range is often better than a fixed list
    learning_rate = trial.suggest_float("learning_rate", 8e-2, 6e-1, log=True)
    schedule_factor = trial.suggest_float("schedule_factor", 0.2, 0.8)
    schedule_patience = trial.suggest_int("schedule_patience", 5, 14)
    
    # #ATTEMPT AT MAKING IT IN REALTION TO THE SCHEDULE
    # patience_center = int(5 + 11 * ((1 - schedule_factor)**2))
    # min_patience = max(5, patience_center - 2)
    # max_patience = min(14, patience_center + 2)
    # schedule_patience = trial.suggest_int("schedule_patience", min_patience, max_patience)

    #optimizer_choice = trial.suggest_categorical("optimizer", ["ASDG", "Adam"])
    init_type = trial.suggest_categorical("init_type", ["random", "small_random"])

    

    n_repeat = 20 # Set to 1 for speed, increase for more robust statistics

    energies_error = []
    epochs_list = []

    for _ in range(n_repeat):

        k_selected=randint(0,number_of_kh-1)
        h_selected=randint(0,number_of_kh-1)
        vqe = VQE(n_wires=n_qubits,
            n_layers=ansatz_depth,
            k=ks[k_selected], h=hs[h_selected], j= 1, 
            shots= n_shots, 
            patience= 30,  
            param_init=init_type)
          
        best_energy, best_epoch, _, _, _= vqe.train_VQE( epochs = 1000,
                                learning_rate = learning_rate,
                                scheduler_patience = schedule_patience,
                                scheduler_factor = schedule_factor,
                                optimizer_choice = "Adam",
                                with_scheduler = True)  
        energies_error.append(float(abs((best_energy - theoretical_energy[h_selected,k_selected]) / theoretical_energy[h_selected,k_selected])))
        epochs_list.append(best_epoch)

    # Calculate means and standard deviation
    mean_enery_error= np.mean(energies_error)
    std_energy = np.std(energies_error)
    mean_epoch = np.mean(epochs_list)

    trial.set_user_attr("my_variable", theoretical_energy[h_selected,k_selected])
    trial.set_user_attr("k_selected", ks[k_selected])
    trial.set_user_attr("h_selected", hs[h_selected])


    return float(mean_enery_error), float(std_energy), float(mean_epoch)

def diagonalize_H(H_matrix):
    """Returns the lowest eigenvector of the Hamiltonian matrix."""
    _, psi = jnp.linalg.eigh(H_matrix)  # Compute eigenvalues and eigenvectors
    return jnp.array(psi[:, 0], dtype=jnp.complex64)  # Return the ground state

n_qubits=6
number_of_kh=10
    # Create meshgrid of the parameter space
    
ks = np.linspace(0, 1, number_of_kh)
hs = np.linspace(0, 2, number_of_kh)
K, H = np.meshgrid(ks, hs)

# Preallocate arrays for Hamiltonian matrices and phase labels.
H_matrices = np.empty((len(ks), len(hs), 2**n_qubits, 2**n_qubits))
theoretical_energy=np.empty((len(ks), len(hs)), dtype=float)

if __name__ == "__main__":

    n_trials=6
    pbar = tqdm(total=n_trials, desc="Optimizing VQE") # Match total to n_trials
    #torch.manual_seed(42)    

    # list_n_qubits=[4, 6, 8, 12]
    # list_ansatz_depth=[2, 4, 6, 9] # paper says 6(9)
    # list_n_shots=[100, 1000, 10000]
    # list_param_init=[None,0,np.pi]
    # list_schedule_patience=[3,5,7,10]
    # list_schedule_factor=[0.1,0.3,0.5,0.7,0.8,0.9]
    # list_learning_rate=[0.001, 0.01, 0.05, 0.07, 0.1]
    # list_optimizers=["ASGD", "Adam"]

    #train_VQE( epochs=100, learning_rate=0.01, scheduler_patience=5, scheduler_factor=0.8, optimizer_choice="Adam", with_scheduler=True)

    def update_pbar(study, trial):
        pbar.update(1)

    storage_url = "sqlite:///vqe_results.db?timeout=60" # Added timeout

    for x, k in enumerate(ks):
        for y, h in enumerate(hs):
            H_matrices[y, x] = np.real(qml.matrix(get_H(n_qubits, k, h))) # Get Hamiltonian matrix

    psis = vmap(vmap(diagonalize_H))(H_matrices)
    
    # Vectorized diagonalization
    for x, k in enumerate(ks):
        for y, h in enumerate(hs):
            theoretical_energy[y, x] = jnp.real(jnp.dot(psis[y, x].conj().T, jnp.dot(H_matrices[y,x], psis[y, x])))
    k_selected=randint(0,number_of_kh-1)
    h_selected=randint(0,number_of_kh-1)

    print(f"k={ks[k_selected]}, h={hs[h_selected]}, {jnp.real(jnp.dot(psis[h_selected, k_selected].conj().T, jnp.dot(H_matrices[h_selected,k_selected], psis[h_selected, k_selected])))}, {theoretical_energy[h_selected,k_selected]} energy matrix: {theoretical_energy}")
    

# 1. PRE-INIT DATABASE SETTINGS
    import sqlite3
    with sqlite3.connect("vqe_results.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=60000;") # 60 seconds

    # 2. CREATE STUDY
    study = optuna.create_study(
        study_name="vqe_search_v5_with_random_kh",
        storage=storage_url,
        directions=["minimize", "minimize", "minimize"],
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )
    
    # Set custom names for objectives
    study.set_metric_names(["Error % mean", "Energy  % Std","Epochs Mean"])

    # 3. OPTIMIZE WITH LIMITED JOBS
    # PennyLane queueing is NOT fully thread-safe. Use n_jobs=1 and parallelize via bash:
    # for i in {1..8}; do python src/random_search_best_optimizer.py & done; wait
    study.optimize(run_trial, n_trials=n_trials, n_jobs=1, callbacks=[update_pbar])

    pbar.close()

    print(f"Total time: {perf_counter() - t1:.2f}s")

    # df = study.trials_dataframe()

    # # Sort by Energy Mean (values_0)
    # df = df.sort_values("values_0") 

    # # Save to CSV for Excel
    # df.to_csv("vqe_optimization_report.csv", index=False)

    # # Ensure column references match the 3 objectives
    # if 'values_2' in df.columns:
    #     print(df[['number', 'values_0', 'values_1', 'values_2', 'params_learning_rate']].head())

    # # Add 3rd target name for the 3D Pareto Front
    # fig = vis.plot_pareto_front(study, target_names=["Energy Mean", "Epochs Mean", "Energy Std"])
    # fig.show()

    # # Importance for Energy Mean
    # fig_energy = vis.plot_param_importances(
    #     study, target=lambda t: t.values[0], target_name="Energy Mean"
    # )
    # fig_energy.show()

    # # Importance for Training Speed (Epochs Mean)
    # fig_speed = vis.plot_param_importances(
    #     study, target=lambda t: t.values[1], target_name="Epochs Mean"
    # )
    # fig_speed.show()
    
    # # Importance for Energy Stability (Standard Deviation)
    # fig_std = vis.plot_param_importances(
    #     study, target=lambda t: t.values[2], target_name="Energy Std"
    # )
    # fig_std.show()