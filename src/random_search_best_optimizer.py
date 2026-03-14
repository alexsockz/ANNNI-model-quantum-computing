from VQE import VQE
import torch
from time import perf_counter
import numpy as np
import optuna
import optuna.visualization as vis
import os

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_trial(trial):
        
    #paper says 6/12
    #n_qubits = trial.suggest_categorical("n_qubits", [4, 6, 8, 12])
    n_qubits=6
    ansatz_depth = trial.suggest_int("ansatz_depth", 2, 9)
    
    #n_shots = trial.suggest_categorical("n_shots", [100, 1000, 10000])
    n_shots=1000
    
    # Suggesting from a range is often better than a fixed list
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    schedule_factor = trial.suggest_float("schedule_factor", 0.1, 0.9)
    schedule_patience = trial.suggest_int("schedule_patience", 3, 10)
    optimizer_choice = trial.suggest_categorical("optimizer", ["ASDG", "Adam"])
    init_type = trial.suggest_categorical("init_type", ["random", "zeros", "pi", "small_random"])

    n_repeat = 10 # Repeat to calculate variance
    energies = []
    epochs_list = []

    for _ in range(n_repeat):
        vqe = VQE(n_wires=n_qubits,
            n_layers=ansatz_depth,
            k=0.2, h=0.5, j= 1, 
            shots= n_shots, 
            patience= 30,  
            param_init=init_type)
          
        best_energy, best_epoch, _, _, _= vqe.train_VQE( epochs = 1000,
                                learning_rate = learning_rate,
                                scheduler_patience = schedule_patience,
                                scheduler_factor = schedule_factor,
                                optimizer_choice = optimizer_choice,
                                with_scheduler = True,
                                optuna_trial=trial)  
        energies.append(best_energy)
        epochs_list.append(best_epoch)

    # Calculate means and standard deviation
    mean_energy = np.mean(energies)
    mean_epoch = np.mean(epochs_list)
    std_energy = np.std(energies)

    return float(mean_energy), float(mean_epoch), float(std_energy)

if __name__ == "__main__":
    torch.manual_seed(42)    

    list_n_qubits=[4, 6, 8, 12]
    list_ansatz_depth=[2, 4, 6, 9] # paper says 6(9)
    list_n_shots=[100, 1000, 10000]
    list_param_init=[None,0,np.pi]
    list_schedule_patience=[3,5,7,10]
    list_schedule_factor=[0.1,0.3,0.5,0.7,0.8,0.9]
    list_learning_rate=[0.001, 0.01, 0.05, 0.07, 0.1]
    list_optimizers=["ASGD", "Adam"]

    #train_VQE( epochs=100, learning_rate=0.01, scheduler_patience=5, scheduler_factor=0.8, optimizer_choice="Adam", with_scheduler=True)
    print("start")
    t1 = perf_counter()

    storage_url = "sqlite:///vqe_results.db"

    study = optuna.create_study(
        study_name="vqe_search",
        storage=storage_url,
        # Update directions to handle 3 objectives: Energy Mean, Epochs Mean, Energy Std Dev
        directions=["minimize", "minimize", "minimize"], 
        pruner=optuna.pruners.MedianPruner(), 
        load_if_exists=True
    )

    study.optimize(run_trial, n_trials=50, n_jobs=1) # parallelization through for i in {1..8}; do python src/random_search_best_optimizer.py & done; wait

    print(f"Total time: {perf_counter() - t1:.2f}s")

    df = study.trials_dataframe()

    # Sort by Energy Mean (values_0)
    df = df.sort_values("values_0") 

    # Save to CSV for Excel
    df.to_csv("vqe_optimization_report.csv", index=False)

    # Ensure column references match the 3 objectives
    if 'values_2' in df.columns:
        print(df[['number', 'values_0', 'values_1', 'values_2', 'params_learning_rate']].head())

    # Add 3rd target name for the 3D Pareto Front
    fig = vis.plot_pareto_front(study, target_names=["Energy Mean", "Epochs Mean", "Energy Std"])
    fig.show()

    # Importance for Energy Mean
    fig_energy = vis.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="Energy Mean"
    )
    fig_energy.show()

    # Importance for Training Speed (Epochs Mean)
    fig_speed = vis.plot_param_importances(
        study, target=lambda t: t.values[1], target_name="Epochs Mean"
    )
    fig_speed.show()
    
    # Importance for Energy Stability (Standard Deviation)
    fig_std = vis.plot_param_importances(
        study, target=lambda t: t.values[2], target_name="Energy Std"
    )
    fig_std.show()