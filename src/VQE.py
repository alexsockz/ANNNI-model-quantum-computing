import pennylane as qml
from torch import optim
from torch import rand, zeros,full,rand, remainder, no_grad, pow, sum, manual_seed, tensor
import torch
import torch.nn as nn
import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from ground_state_at_borders import calc_state

class VQE:
    def __init__(self, n_wires, n_layers, k, h, j=1, shots=1000, patience=30, param_init:str="random", noise=False): #supervised=true
        self.n = n_wires
        self.m = n_layers
        self.k = k
        self.h = h
        self.j = j
        self.noise = noise
        self.early_stopping_patience = patience
        self.shots = shots
        
        # 1. Device Selection for better performances depending on available devices on computer
        #qubit = cpu
        if self.noise==True:
            noise_model = NoiseModel()
            error = depolarizing_error(0.01, 1)  # 1-qubit depolarizing
            noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

            self.dev = qml.device("qiskit.aer", wires=self.n, noise_model=noise_model)
        elif n_wires < 15:
            self.dev = qml.device("lightning.qubit", wires=self.n)
        else:
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n)
            except Exception:
                self.dev = qml.device("lightning.qubit", wires=self.n)
        
        # 2. Setup the QNode ONCE during init
        # Using adjoint for lightning is much faster if shots=None, 
        # this is because it Computes gradients analytically using the adjoint (reverse) of the quantum circuit's unitary operator, but it's unrealistic
        # but for finite shots, parameter-shift is correct.
        # Computes gradients numerically using finite differences
        # For each parameter, it evaluates the circuit twice: once at θ+π/2 and once at θ-π/2
        # Much slower but realistic hardware simulation with shot-based measurements
        if shots==None:
            self.qnode = qml.QNode(self._train_circuit, self.dev, interface="torch", diff_method="best")
        else:    
            self.qnode = qml.QNode(self._train_circuit, self.dev, interface="torch", diff_method="parameter-shift")
        self.qnode._set_shots(self.shots)

        # 3. Param Init
        # machine learning stuff, choose a way to start the random weights, zeros and small random
        if param_init=="random":
            self.parameters_vqe = nn.Parameter(rand((self.m, self.n)) * 2 * np.pi, requires_grad=True)
        elif param_init=="zeros":
            self.parameters_vqe = nn.Parameter(zeros((self.m, self.n)), requires_grad=True)
        elif param_init=="pi":
            self.parameters_vqe = nn.Parameter(full((self.m, self.n),np.pi), requires_grad=True)
        elif param_init=="small_random":
            self.parameters_vqe = nn.Parameter(rand((self.m, self.n)) * 0.01, requires_grad=True)

        # Pre-calculate bitstrings for the 2^n states in bigendian order
        self.bitstrings = tensor([[(i >> (self.n - 1 - j)) & 1 for j in range(self.n)] for i in range(2**self.n)], dtype=torch.float64)

        # Pre-calculate eigenvalues for all Hamiltonian terms to make training extremely fast
        self.eigenvalues_nn = []
        for i in range(self.n - 1):
            self.eigenvalues_nn.append(pow(-1.0, sum(self.bitstrings[:, [i, i+1]], dim=1)))
            
        self.eigenvalues_nnn = []
        for i in range(self.n - 2):
            self.eigenvalues_nnn.append(pow(-1.0, sum(self.bitstrings[:, [i, i+2]], dim=1)))
            
        self.eigenvalues_z = []
        for i in range(self.n):
            self.eigenvalues_z.append(pow(-1.0, sum(self.bitstrings[:, [i]], dim=1)))

    # Using 'probs' is often more stable for gradients than 'counts', 
    # but it represents the exact same hardware reality (sampling). 
    # using expvalue is irrealistic since you don't have access to the computed matrix   
    def _train_circuit(self, params, basis="Z",starting_state=None):

        if(starting_state!=None):
            qml.StatePrep(starting_state, wires=range(self.n), normalize = True)
        # The ansatz must be purely gate operations
        self.ansatz(params)
        
        # Apply basis rotation: Hadamard to measure in X basis
        if basis == "X":
            for i in range(self.n): 
                qml.Hadamard(wires=i)
        
        # Measure in computational basis after basis rotation
        return qml.probs(wires=range(self.n))        
    
    def get_phase(self,k, h):
        """Get the phase from the DMRG transition lines"""
        # If under the Ising Transition Line (Left side)
        if k < .5 and h < self.ising_transition(k):
            return 0 # Ferromagnetic
        # If under the Kosterlitz-Thouless Transition Line (Right side)

        elif k > .5 and h < self.kt_transition(k):
            return 1 # Antiphase
        return 2 # else i
    

    def kt_transition(self,k):
        """Kosterlitz-Thouless transition line"""
        return 1.05 * np.sqrt((k - 0.5) * (k - 0.1))

    def ising_transition(self,k):
        """Ising transition line"""
        return np.where(k == 0, 1, (1 - k) * (1 - np.sqrt((1 - 3 * k + 4 * k**2) / (1 - k))) / np.maximum(k, 1e-9))

    def bkt_transition(self,k):
        """Floating Phase transition line"""
        return 1.05 * (k - 0.5)

    def train_VQE(self, epochs=3000, learning_rate=0.151315, scheduler_patience=12, scheduler_factor=0.75816, optimizer_choice="Adam", with_scheduler=True, non_zero_state=False):
        
        phase=self.get_phase(self.k, self.h)

        starting_state=None
        if (non_zero_state==True):
            if(phase==0):
                starting_state,phase2=calc_state(self.n,0,0)
                print(f"{phase}{phase2}")
            elif(phase==1):
                starting_state,phase2=calc_state(self.n,1,0)
                print(f"{phase}{phase2}")
            else:
                starting_state,phase2=calc_state(self.n,0,2)
                print(f"{phase}{phase2}")
            starting_state=torch.tensor(starting_state,dtype=torch.complex128)

        # various variables to keep track of what is happening to the model
        best_energy = float('inf')
        best_params = self.parameters_vqe.detach().clone()
        patience_counter = 0
        best_epoch = epochs
        last_epoch = epochs

        energy_history = []
        lr_history = []

        # Averaged Stochastic Gradient Descent does not need a scheduling for learning rate
        scheduler_decision = with_scheduler and optimizer_choice != "ASGD"

        if optimizer_choice == "ASGD": # Averaged Stochastic Gradient Descent 
            optimizer = optim.ASGD([self.parameters_vqe], lr=learning_rate, weight_decay=0)
        else: # Adaptive Moment Estimation (the best one)
            optimizer = optim.Adam([self.parameters_vqe], lr=learning_rate, weight_decay=0)
        
        # during training it might happen that the pit you want to fall in 
        # gets too small for the learning rate to let you jump to a lower place
        # this is solvable by making the learning rate smaller,
        # a scheduler reduce the learning rate depending on various parameters
        # this scheduler, after "patiance" number of steps, "looses it's patiance"
        # and reduce the learning rate by multipling the rate by the "factor"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                         patience=scheduler_patience if scheduler_decision else 1000, 
                                                         factor=scheduler_factor if scheduler_decision else 1.0)
        # now the training begins
        for epoch in range(epochs):
            # Delete old gradients
            optimizer.zero_grad()
            
            # Use the pre-initialized qnode to measure in both Z and X bases (with shots)
            probs_z = self.qnode(self.parameters_vqe, basis="Z", starting_state=starting_state)
            probs_x = self.qnode(self.parameters_vqe, basis="X", starting_state=starting_state)
            
            # Compute energy from shot-based probabilities
            energy = self._compute_energy_from_probs(probs_z, probs_x)
            energy_val = energy.item()
            
            # # --- Optuna Reporting & Pruning ---
            # doesn't work for multiobjective ??
            # if optuna_trial is not None:
            #     optuna_trial.report(energy_val, step=epoch)
            #     if optuna_trial.should_prune():
            #         raise optuna.exceptions.TrialPruned()

            # remember the best model
            if energy_val < best_energy:
                best_energy = energy_val
                best_params = self.parameters_vqe.detach().clone()
                patience_counter = 0
                best_epoch = epoch
            else:
                # if i don't get better this epoch add to the counter
                patience_counter += 1

            # if too much time elapses without getting better lets stop
            if patience_counter >= self.early_stopping_patience:
                last_epoch=epoch
                break
            
            energy_history.append(energy_val)
            lr_history.append(optimizer.param_groups[0]['lr'])

            # now that i have a gradient i propagate the change to all the weights used to calculate the energy
            # this is basically black magic, torch is magic
            energy.backward()
            optimizer.step()

            # if i'm using the scheduler do this
            if scheduler_decision:
                scheduler.step(energy_val)

            # no grad Turns off PyTorch's autograd engine (no backpropagation tracking)
            # Saves memory and computation time
            # Tensors created/modified inside won't track computational history
            # During training, you want gradients only for optimizer.step() and energy.backward()
            # Operations that don't affect training (like parameter cleanup or inference) should skip gradient overhead
            # Saves memory and speeds up non-learning operations

            with no_grad():
                # Keep parameters in [0, 2pi]
                # might be wrong?? TODO check
                self.parameters_vqe.copy_(remainder(self.parameters_vqe, 2 * np.pi))

        with no_grad():
            self.parameters_vqe.copy_(best_params)

        return best_energy, best_epoch, last_epoch, energy_history, lr_history

    def _compute_energy_from_probs(self, probs_z, probs_x):
        """
        Calculates expectation values from shot-based probabilities.
        Realistic hardware simulation: uses finite-shot measurement outcomes.
        """
        probs_z = probs_z.float()
        probs_x = probs_x.float()
        
        energy = tensor(0.0, dtype=torch.float64)

        # Nearest Neighbor XX interactions
        for i in range(self.n - 1):
            energy -= self.j * sum(self.eigenvalues_nn[i] * probs_x)
        
        # Next-Nearest Neighbor XX interactions
        for i in range(self.n - 2):
            energy += self.k * sum(self.eigenvalues_nnn[i] * probs_x)
            
        # Transverse Field Z interactions
        for i in range(self.n):
            energy -= self.h * sum(self.eigenvalues_z[i] * probs_z)

        return energy
    
    def ansatz(self, params):
        # ry O     X
        # ry X O
        # ry   X O
        # ry     X O
        for j in range(self.m):
            for i in range(self.n):
                qml.RY(params[j, i], wires=i)
            for i in range(self.n):
                qml.CNOT(wires=[i, (i + 1) % self.n])

if __name__ == "__main__":
    from time import perf_counter

    # manual_seed(42)
    print("start")
    t1 = perf_counter()
    n_qubits=6
    k=0.5
    h=0.5
    #manual_seed(42)
    # Note: For 2 qubits, next-nearest neighbor (k) doesn't exist, which is fine.
    vqe = VQE(n_wires=n_qubits, n_layers=9, k=k, h=h, shots=None, noise=False)  
    best_energy, best_epoch, last_epoch, energy_history, lr_history = vqe.train_VQE(non_zero_state=False)  

    import energy
    print(energy.theoretical_energy(n_qubits,k,h))
    import matplotlib.pyplot as plt

    plt.plot(energy_history)
    plt.savefig("qve.png")
    print(f"\nFinal Ground State Energy: {best_energy:.6f}")
    print(f"Total time: {perf_counter() - t1:.2f}s")