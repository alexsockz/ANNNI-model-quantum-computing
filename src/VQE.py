import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import warnings

# Ignore specific PennyLane shot warnings for cleaner output
warnings.filterwarnings("ignore", message="Setting shots on device is deprecated")

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class VQE:
    def __init__(self, n_wires, n_layers, k, h, j=1):
        self.n = n_wires
        self.m = n_layers
        self.k = k
        self.h = h
        self.j = j
        self.early_stopping_patience=30
        self.shots = 1000
        self.parameters_vqe = nn.Parameter(torch.rand((self.m, self.n)) * 2 * np.pi, requires_grad=True)

    def train_VQE(self, epochs=100, learning_rate=0.01):
        dev = qml.device("default.qubit", wires=self.n, shots=self.shots)

        # Using 'probs' is often more stable for gradients than 'counts', 
        # but it represents the exact same hardware reality (sampling).
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(params, basis="Z"):
            self.ansatz(params)
            if basis == "X":
                for i in range(self.n): qml.Hadamard(wires=i)
            # return qml.probs() gives a tensor of shape (2^n,)
            # This is functionally identical to counts/shots
            return qml.probs(wires=range(self.n))

        optimizer = torch.optim.Adam([self.parameters_vqe], lr=learning_rate, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8)

        best_energy = float('inf')
        best_params = self.parameters_vqe.detach().clone()
        patience_counter=0

        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 1. Get probabilities (the "normalized counts")
            probs_z = circuit(self.parameters_vqe, basis="Z")
            probs_x = circuit(self.parameters_vqe, basis="X")
            
            # 2. Compute energy using ONLY Torch ops
            energy = self.compute_energy_from_probs(probs_z, probs_x)
            
            energy.backward()
            optimizer.step()
            scheduler.step(energy)
            with torch.no_grad():
                self.parameters_vqe.copy_(torch.remainder(self.parameters_vqe, 2 * np.pi))

            energy_val = energy.item()
            
            if energy_val < best_energy:
                best_energy = energy_val
                # Usiamo .clone() per evitare che i parametri salvati cambino 
                # quando l'ottimizzatore aggiorna quelli attuali
                best_params = self.parameters_vqe.detach().clone()
                print(f"Nuovo minimo trovato all'epoca {epoch}: {best_energy:.6f}")
                patience_counter = 0 # Reset del counter perché abbiamo migliorato
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"\n--- Early Stopping all'epoca {epoch} ---")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Energy = {energy.item():.6f}")
        with torch.no_grad():
            self.parameters_vqe.copy_(best_params)
        return best_energy

    def compute_energy_from_probs(self, probs_z, probs_x):
        """
        Calculates expectation values from the probability tensor.
        This replaces your counts_to_expectation loop.
        """
        
        energy = torch.tensor(0.0)

        # Pre-calculate bitstrings for the 2^n states
        # Example for 2 qubits: [[0,0], [0,1], [1,0], [1,1]]
        states = torch.tensor([[int(b) for b in format(i, f'0{self.n}b')] for i in range(2**self.n)])

        def get_expectation(probs, indices):
            # Calculate parity: (-1)^sum(bits at indices)
            # This is the vectorized version of your eigenvalue loop
            relevant_bits = states[:, indices]
            parities = torch.pow(-1, torch.sum(relevant_bits, dim=1))
            return torch.sum(parities * probs)

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
    print("start")
    t1 = perf_counter()
    # Note: For 2 qubits, next-nearest neighbor (k) doesn't exist, which is fine.
    vqe = VQE(n_wires=4, n_layers=2, k=0.2, h=0.5)  
    energy = vqe.train_VQE(epochs=300, learning_rate=0.05)  
    
    print(f"\nFinal Ground State Energy: {energy:.6f}")
    print(f"Total time: {perf_counter() - t1:.2f}s")