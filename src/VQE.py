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
        
        # 1. Device Selection for better performances depending on available devices on computer
        #qubit = cpu
        if n_wires < 15:
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
    
    # Using 'probs' is often more stable for gradients than 'counts', 
    # but it represents the exact same hardware reality (sampling). 
    # using expvalue is irrealistic since you don't have access to the computed matrix   
    def _train_circuit(self, params, basis="Z"):
        # The ansatz must be purely gate operations
        self.ansatz(params)
        
        # this is to measure in base X instead of Z while the measurment remain the same
        if basis == "X":
            for i in range(self.n): 
                qml.Hadamard(wires=i)
        
        # In Pennylane, all ops (ansatz + basis change) must be queued BEFORE measurements, 
        # this makes parallelization a bit tricky because wires might get crossed between threads
        return qml.probs(wires=range(self.n))        
        
    def train_VQE(self, epochs=100, learning_rate=0.01, scheduler_patience=5, scheduler_factor=0.8, optimizer_choice="Adam", with_scheduler=True, optuna_trial=None):
        
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
            # ignore this, says 
            # "delete the old gradients calculated, don't add it to the one i'm calculating this epoch"
            optimizer.zero_grad()
            
            # Use the pre-initialized qnode
            probs_z = self.qnode(self.parameters_vqe, basis="Z")
            probs_x = self.qnode(self.parameters_vqe, basis="X")
            
            # basically calculate the hamiltonian:
            energy = self._compute_energy_from_probs(probs_z, probs_x)
            #i'm using torch, so everything is a tensor            
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
        Calculates expectation values from the probability tensor.
        This replaces your counts_to_expectation loop.
        """
        
        energy = tensor(0.0)

        # Pre-calculate bitstrings for the 2^n states
        # the eigenvalue will be positive or negative depending on the ammount bits set
        # Example for 2 qubits: [[0,0], [0,1], [1,0], [1,1]]
        states = tensor([[int(b) for b in format(i, f'0{self.n}b')] for i in range(2**self.n)])

        def get_expectation(probs, indices):
            # Calculate parity: (-1)^sum(bits at indices) (using what we did before )
            # This is the vectorized version of your eigenvalue loop
            relevant_bits = states[:, indices] # either i + it's neighbour or i + it's second neighbour
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

    manual_seed(42)
    print("start")
    t1 = perf_counter()
    # Note: For 2 qubits, next-nearest neighbor (k) doesn't exist, which is fine.
    vqe = VQE(n_wires=4, n_layers=2, k=0.2, h=0.5)  
    best_energy, best_epoch, last_epoch, energy_history, lr_history = vqe.train_VQE(epochs=300, learning_rate=0.05)  
    
    print(f"\nFinal Ground State Energy: {best_energy:.6f}")
    print(f"Total time: {perf_counter() - t1:.2f}s")