import pennylane as qml
from torch import optim
from torch import rand, zeros,full,rand, remainder, no_grad, pow, sum, manual_seed, tensor
import torch.nn as nn
import numpy as np

class VQE:
    def __init__(self, n_wires, n_layers, k, h, j=1, shots=1000, patience=30, param_init:str="random"):
        self.n = n_wires
        self.m = n_layers
        self.k = k
        self.h = h
        self.j = j
        self.early_stopping_patience = patience
        self.shots = shots
        
        # 1. Device Selection (Shots are set here globally)
        if n_wires < 15:
            self.dev = qml.device("lightning.qubit", wires=self.n)
        else:
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n)
            except Exception:
                self.dev = qml.device("lightning.qubit", wires=self.n)
        
        # 2. Setup the QNode ONCE during init
        # Using adjoint for lightning is much faster if shots=None, 
        # but for finite shots, parameter-shift is correct.
        self.qnode = qml.QNode(self._train_circuit, self.dev, interface="torch", diff_method="parameter-shift")
        self.qnode._set_shots(self.shots)

        # 3. Param Init
        if param_init=="random":
            self.parameters_vqe = nn.Parameter(rand((self.m, self.n)) * 2 * np.pi, requires_grad=True)
        elif param_init=="zeros":
            self.parameters_vqe = nn.Parameter(zeros((self.m, self.n)), requires_grad=True)
        elif param_init=="pi":
            self.parameters_vqe = nn.Parameter(full((self.m, self.n),np.pi), requires_grad=True)
        elif param_init=="small_random":
            self.parameters_vqe = nn.Parameter(rand((self.m, self.n)) * 0.01, requires_grad=True)
    
    # Using 'probs' is often more stable for gradients than 'counts', 
    # but it represents the exact same hardware reality (sampling).    
    def _train_circuit(self, params, basis="Z"):
        # The ansatz must be purely gate operations
        self.ansatz(params)
        
        # Rotation for X basis must happen BEFORE the return
        if basis == "X":
            for i in range(self.n): 
                qml.Hadamard(wires=i)
        
        # In Pennylane, all ops (ansatz + basis change) must be queued BEFORE measurements
        return qml.probs(wires=range(self.n))        
        
    def train_VQE(self, epochs=100, learning_rate=0.01, scheduler_patience=5, scheduler_factor=0.8, optimizer_choice="Adam", with_scheduler=True, optuna_trial=None):
        
        best_energy = float('inf')
        best_params = self.parameters_vqe.detach().clone()
        patience_counter = 0
        best_epoch = epochs
        last_epoch = epochs

        energy_history = []
        lr_history = []

        # Fix the typo check: ASGD
        scheduler_decision = with_scheduler and optimizer_choice != "ASGD"

        if optimizer_choice == "ASGD":
            optimizer = optim.ASGD([self.parameters_vqe], lr=learning_rate, weight_decay=0)
        else:
            optimizer = optim.Adam([self.parameters_vqe], lr=learning_rate, weight_decay=0)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                         patience=scheduler_patience if scheduler_decision else 1000, 
                                                         factor=scheduler_factor if scheduler_decision else 1.0)

        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Use the pre-initialized qnode
            probs_z = self.qnode(self.parameters_vqe, basis="Z")
            probs_x = self.qnode(self.parameters_vqe, basis="X")
            
            energy = self._compute_energy_from_probs(probs_z, probs_x)            
            energy_val = energy.item()
            
            # # --- Optuna Reporting & Pruning ---
            # if optuna_trial is not None:
            #     optuna_trial.report(energy_val, step=epoch)
            #     if optuna_trial.should_prune():
            #         raise optuna.exceptions.TrialPruned()

            if energy_val < best_energy:
                best_energy = energy_val
                best_params = self.parameters_vqe.detach().clone()
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                last_epoch=epoch
                break
            
            energy_history.append(energy_val)
            lr_history.append(optimizer.param_groups[0]['lr'])

            energy.backward()
            optimizer.step()

            if scheduler_decision:
                scheduler.step(energy_val)

            with no_grad():
                # Keep parameters in [0, 2pi]
                self.parameters_vqe.copy_(remainder(self.parameters_vqe, 2 * np.pi))

        with no_grad():
            self.parameters_vqe.copy_(best_params)

        return best_energy, best_epoch, last_epoch, energy_history, lr_history

    def _compute_energy_from_probs(self, probs_z, probs_x):
        """
        Calculates expectation values from the probability tensor.
        This replaces your counts_to_expectation loop.
        """
        
        energy = tensor(0.0)

        # Pre-calculate bitstrings for the 2^n states
        # Example for 2 qubits: [[0,0], [0,1], [1,0], [1,1]]
        states = tensor([[int(b) for b in format(i, f'0{self.n}b')] for i in range(2**self.n)])

        def get_expectation(probs, indices):
            # Calculate parity: (-1)^sum(bits at indices)
            # This is the vectorized version of your eigenvalue loop
            relevant_bits = states[:, indices]
            parities = pow(-1, sum(relevant_bits, dim=1))
            return sum(parities * probs)

        # Nearest Neighbor ZZ
        for i in range(self.n - 1):
            energy -= self.j * get_expectation(probs_z, [i, i+1])
        
        # Next-Nearest Neighbor ZZ
        for i in range(self.n - 2):
            energy += self.k * get_expectation(probs_z, [i, i+2])
            
        # Transverse Field X
        for i in range(self.n):
            energy -= self.h * get_expectation(probs_x, [i])

        return energy

    def ansatz(self, params):
        for j in range(self.m):
            for i in range(self.n):
                qml.RY(params[j, i], wires=i)
            for i in range(self.n):
                qml.CNOT(wires=[i, (i + 1) % self.n])

if __name__ == "__main__":
    from time import perf_counter

    manual_seed(42)
    print("start")
    t1 = perf_counter()
    # Note: For 2 qubits, next-nearest neighbor (k) doesn't exist, which is fine.
    vqe = VQE(n_wires=4, n_layers=2, k=0.2, h=0.5)  
    best_energy, best_epoch, last_epoch, energy_history, lr_history = vqe.train_VQE(epochs=300, learning_rate=0.05)  
    
    print(f"\nFinal Ground State Energy: {best_energy:.6f}")
    print(f"Total time: {perf_counter() - t1:.2f}s")