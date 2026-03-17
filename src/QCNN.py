import pennylane as qml
import numpy as np
import torch.nn as nn
from torch import clip, rand, zeros,full,rand, remainder, no_grad, pow, sum, manual_seed, tensor
from torch import optim
import VQE
class QCNN:
    def __init__(self, num_qubits,shots,param_init:str="random") -> None:
        self.shots=shots
        self.num_qubits=num_qubits
        self.vqe=VQE()
        if param_init=="random":
            self.parameters_vqe = nn.Parameter(rand((self.m, self.n)) * 2 * np.pi, requires_grad=True)
        elif param_init=="zeros":
            self.parameters_vqe = nn.Parameter(zeros((self.m, self.n)), requires_grad=True)
        elif param_init=="pi":
            self.parameters_vqe = nn.Parameter(full((self.m, self.n),np.pi), requires_grad=True)
        elif param_init=="small_random":
            self.parameters_vqe = nn.Parameter(rand((self.m, self.n)) * 0.01, requires_grad=True)
            

        if num_qubits < 15:
            self.dev = qml.device("lightning.qubit", wires=self.n)
        else:
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n)
            except Exception:
                self.dev = qml.device("lightning.qubit", wires=self.n)
        
        self.qnode = qml.QNode(self.qcnn_circuit, self.dev, interface="torch", diff_method="parameter-shift")
        self.qnode._set_shots(self.shots)


    def qcnn_ansatz(self,params):
        """Ansatz of the QCNN model
        Repetitions of the convolutional and pooling blocks
        until only 2 wires are left unmeasured
        """

        # Convolution block
        def conv(wires, params, index):
            if len(wires) % 2 == 0:
                groups = wires.reshape(-1, 2)
            else:
                groups = wires[:-1].reshape(-1, 2)
                qml.RY(params[index], wires=int(wires[-1]))
                index += 1

            for group in groups:
                qml.CNOT(wires=[int(group[0]), int(group[1])])
                for wire in group:
                    qml.RY(params[index], wires=int(wire))
                    index += 1

            return index

        # Pooiling block
        def pool(wires, params, index):
            # Process wires in pairs: measure one and conditionally rotate the other.
            for wire_pool, wire in zip(wires[0::2], wires[1::2]):
                m_0 = qml.measure(int(wire_pool))
                qml.cond(m_0 == 0, qml.RX)(params[index],     wires=int(wire))
                qml.cond(m_0 == 1, qml.RX)(params[index + 1], wires=int(wire))
                index += 2
                # Remove the measured wire from active wires.
                wires = np.delete(wires, np.where(wires == wire_pool))

            # If an odd wire remains, apply a RX rotation.
            if len(wires) % 2 != 0:
                qml.RX(params[index], wires=int(wires[-1]))
                index += 1

            return index, wires

        # Initialize active wires and parameter index.
        active_wires = np.arange(self.num_qubits)
        index = 0

        # Initial layer: apply RY to all wires.
        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            index += 1

        # Repeatedly apply convolution and pooling until there are 2 unmeasured wires
        while len(active_wires) > 2:
            # Convolution
            index = conv(active_wires, params, index)
            # Pooling
            index, active_wires = pool(active_wires, params, index)
            qml.Barrier()

        # Final layer: apply RY to the remaining active wires.
        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            index += 1

        return index, active_wires
    
    def qcnn_circuit(self, params, state):
        """QNode with QCNN ansatz and probabilities of unmeasured qubits as output"""
        # Input ground state from diagonalization
        qml.StatePrep(state, wires=range(self.num_qubits), normalize = True)
        # QCNN
        _, output_wires = self.qcnn_ansatz(params)

        return qml.probs([int(k) for k in output_wires])
    
    def cross_entropy(self,pred, Y, T):
        """
        Multi-class cross entropy loss function
        """
        epsilon = 1e-9  # Small value for numerical stability
        pred = clip(pred, epsilon, 1 - epsilon)  # Prevent log(0)

        # Apply sharpening (raise probabilities to the power of 1/T)
        pred_sharpened = pred ** (1 / T)
        pred_sharpened /= sum(pred_sharpened, axis=1, keepdims=True)  # Re-normalize

        loss = -sum(Y * log(pred_sharpened), axis=1)
        return mean(loss)

    def train_qcnn(self, epochs=100, learning_rate=0.01, scheduler_patience=5, scheduler_factor=0.8, optimizer_choice="Adam", with_scheduler=True, optuna_trial=None):
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

