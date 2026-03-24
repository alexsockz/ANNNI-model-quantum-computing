import os
import warnings

# Disable CUDA entirely to avoid CuDNN version mismatch error
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_QUIET_STARTUP"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import pennylane as qml
import numpy as np
from jax import jit, vmap, value_and_grad, random, config
from jax import numpy as jnp
import optax

from pennylane import numpy as pnp
from pennylane import DepolarizingChannel

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import multiprocessing as mp
from functools import partial

config.update("jax_enable_x64", True)

seed = 123456

# Setting our constants
num_qubits = 8 # Number of spins in the Hamiltonian (= number of qubits)
side = 20     # Discretization of the Phase Diagram

# Note: noise is now handled per training process, see train_qcnn_for_noise function


def get_H(num_spins, k, h):
    """Construction function the ANNNI Hamiltonian (J=1)"""

    # Interaction between spins (neighbouring):
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H  - (qml.PauliX(i) @ qml.PauliX(i + 1))

    # Interaction between spins (next-neighbouring):
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))

    # Interaction of the spins with the magnetic field
    for i in range(0, num_spins):
        H = H - h * qml.PauliZ(i)

    return H

def kt_transition(k):
    """Kosterlitz-Thouless transition line"""
    return 1.05 * np.sqrt((k - 0.5) * (k - 0.1))

def ising_transition(k):
    """Ising transition line"""
    return np.where(k == 0, 1, (1 - k) * (1 - np.sqrt((1 - 3 * k + 4 * k**2) / (1 - k))) / np.maximum(k, 1e-9))

def bkt_transition(k):
    """Floating Phase transition line"""
    return 1.05 * (k - 0.5)

def get_phase(k, h):
    """Get the phase from the DMRG transition lines"""
    # If under the Ising Transition Line (Left side)
    if k < .5 and h < ising_transition(k):
        return 0 # Ferromagnetic
    # If under the Kosterlitz-Thouless Transition Line (Right side)

    elif k > .5 and h < kt_transition(k):
        return 1 # Antiphase
    return 2 # else it is Paramagnetic

def diagonalize_H(H_matrix):
    """Returns the lowest eigenvector of the Hamiltonian matrix."""
    _, psi = jnp.linalg.eigh(H_matrix)  # Compute eigenvalues and eigenvectors
    return jnp.array(psi[:, 0], dtype=jnp.complex64)  # Return the ground state

# Create meshgrid of the parameter space
ks = np.linspace(0, 1, side)
hs = np.linspace(0, 2, side)
K, H = np.meshgrid(ks, hs)

# Preallocate arrays for Hamiltonian matrices and phase labels.
H_matrices = np.empty((len(ks), len(hs), 2**num_qubits, 2**num_qubits))
phases = np.empty((len(ks), len(hs)), dtype=int)

for x, k in enumerate(ks):
    for y, h in enumerate(hs):
        H_matrices[y, x] = np.real(qml.matrix(get_H(num_qubits, k, h))) # Get Hamiltonian matrix
        phases[y, x] = get_phase(k, h)  # Get the respective phase given k and h

# Vectorized diagonalization
psis = vmap(vmap(diagonalize_H))(H_matrices)


def get_num_params(num_qubits):
    """Compute the number of parameters for QCNN ansatz"""
    def qcnn_ansatz_temp(num_qubits, params):
        def conv(wires, params, index):
            if len(wires) % 2 == 0:
                groups = wires.reshape(-1, 2)
            else:
                groups = wires[:-1].reshape(-1, 2)
                index += 1
            for group in groups:
                for wire in group:
                    index += 1
            return index
        
        def pool(wires, params, index):
            for wire_pool, wire in zip(wires[0::2], wires[1::2]):
                index += 2
                wires = np.delete(wires, np.where(wires == wire_pool))
            if len(wires) % 2 != 0:
                index += 1
            return index, wires
        
        active_wires = np.arange(num_qubits)
        index = 0
        for wire in active_wires:
            index += 1
        while len(active_wires) > 2:
            index = conv(active_wires, params, index)
            index, active_wires = pool(active_wires, params, index)
        for wire in active_wires:
            index += 1
        return index, active_wires
    
    return qcnn_ansatz_temp(num_qubits, [0]*100)[0]

num_params = get_num_params(num_qubits)


def cross_entropy(pred, Y, T):
    """Multi-class cross entropy loss function"""
    epsilon = 1e-9  # Small value for numerical stability
    pred = jnp.clip(pred, epsilon, 1 - epsilon)  # Prevent log(0)

    # Apply sharpening (raise probabilities to the power of 1/T)
    pred_sharpened = pred ** (1 / T)
    pred_sharpened /= jnp.sum(pred_sharpened, axis=1, keepdims=True)  # Re-normalize

    loss = -jnp.sum(Y * jnp.log(pred_sharpened), axis=1)
    return jnp.mean(loss)

# Mask for the analytical points
analytical_mask = (K == 0) | (H == 0)

def train_qcnn_for_noise(noise_value, num_epochs=100, lr=5e-2, T=.5, seed=seed, batch_size=16):
    """Train QCNN for a specific noise level.
    
    Args:
        noise_value: Noise strength (None for no noise, or float between 0-1)
        num_epochs: Number of training epochs
        lr: Learning rate
        T: Temperature parameter
        seed: Random seed
        batch_size: Mini-batch size
    
    Returns:
        dict with noise_value, trained_params, loss_curve
    """
    # Set answer based on noise value
    answer_local = "n" if noise_value is None else "y"
    noise_strength_local = noise_value if noise_value is not None else None
    
    # Create device for this process
    if answer_local == "y":
        dev = qml.device("default.mixed", wires=num_qubits)
    else:
        dev = qml.device("default.qubit", wires=num_qubits)

    # Define local qcnn_ansatz without noise
    def qcnn_ansatz_local(num_qubits, params):
        """Ansatz of the QCNN model without noise"""
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

        def pool(wires, params, index):
            for wire_pool, wire in zip(wires[0::2], wires[1::2]):
                m_0 = qml.measure(int(wire_pool))
                qml.cond(m_0 == 0, qml.RX)(params[index],     wires=int(wire))
                qml.cond(m_0 == 1, qml.RX)(params[index + 1], wires=int(wire))
                index += 2
                wires = np.delete(wires, np.where(wires == wire_pool))

            if len(wires) % 2 != 0:
                qml.RX(params[index], wires=int(wires[-1]))
                index += 1
            return index, wires

        active_wires = np.arange(num_qubits)
        index = 0

        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            index += 1

        while len(active_wires) > 2:
            index = conv(active_wires, params, index)
            index, active_wires = pool(active_wires, params, index)
            qml.Barrier()

        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            index += 1

        return index, active_wires

    # Define local qcnn_circuit using clean ansatz (used for training)
    @qml.qnode(qml.device("default.qubit", wires=num_qubits))
    def qcnn_circuit_local(params, state):
        qml.StatePrep(state, wires=range(num_qubits), normalize=True)
        _, output_wires = qcnn_ansatz_local(num_qubits, params)
        return qml.probs([int(k) for k in output_wires])

    # Define noisy ansatz
    def qcnn_ansatz_noisy_local(num_qubits, params):
        """Ansatz of the QCNN model with noise injection"""
        def conv(wires, params, index):
            if len(wires) % 2 == 0:
                groups = wires.reshape(-1, 2)
            else:
                groups = wires[:-1].reshape(-1, 2)
                qml.RY(params[index], wires=int(wires[-1]))
                if answer_local == "y":
                    qml.DepolarizingChannel(noise_strength_local, wires=int(wires[-1]))
                index += 1

            for group in groups:
                qml.CNOT(wires=[int(group[0]), int(group[1])])
                if answer_local == "y":
                    qml.DepolarizingChannel(noise_strength_local, wires=int(group[0]))
                    qml.DepolarizingChannel(noise_strength_local, wires=int(group[1]))
                for wire in group:
                    qml.RY(params[index], wires=int(wire))
                    if answer_local == "y":
                        qml.DepolarizingChannel(noise_strength_local, wires=int(wire))
                    index += 1
            return index

        def pool(wires, params, index):
            for wire_pool, wire in zip(wires[0::2], wires[1::2]):
                m_0 = qml.measure(int(wire_pool))
                qml.cond(m_0 == 0, qml.RX)(params[index], wires=int(wire))
                qml.cond(m_0 == 1, qml.RX)(params[index + 1], wires=int(wire))
                if answer_local == "y":
                    qml.DepolarizingChannel(noise_strength_local, wires=int(wire))
                index += 2
                wires = np.delete(wires, np.where(wires == wire_pool))

            if len(wires) % 2 != 0:
                qml.RX(params[index], wires=int(wires[-1]))
                if answer_local == "y":
                    qml.DepolarizingChannel(noise_strength_local, wires=int(wires[-1]))
                index += 1
            return index, wires

        active_wires = np.arange(num_qubits)
        index = 0

        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            if answer_local == "y":
                qml.DepolarizingChannel(noise_strength_local, wires=int(wire))
            index += 1

        while len(active_wires) > 2:
            index = conv(active_wires, params, index)
            index, active_wires = pool(active_wires, params, index)
            qml.Barrier()

        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            if answer_local == "y":
                qml.DepolarizingChannel(noise_strength_local, wires=int(wire))
            index += 1
        return index, active_wires

    @qml.qnode(dev)
    def qcnn_noisy_local(params, state):
        qml.StatePrep(state, wires=range(num_qubits), normalize=True)
        _, output_wires = qcnn_ansatz_noisy_local(num_qubits, params)
        return qml.probs([int(k) for k in output_wires])

    # Vectorized circuits
    vectorized_qcnn_circuit_local = vmap(jit(qcnn_circuit_local), in_axes=(None, 0))
    vectorized_qcnn_noisy_local = vmap(jit(qcnn_noisy_local), in_axes=(None, 0))

    # Training function
    def train_qcnn_local(use_noisy=False):
        """Train QCNN with optional noise"""
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)

        circuit_to_use = vectorized_qcnn_noisy_local if use_noisy else vectorized_qcnn_circuit_local

        def loss_fun(params, X, Y):
            preds = circuit_to_use(params, X)
            return cross_entropy(preds, Y, T)

        X_train, Y_train = psis[analytical_mask], phases[analytical_mask]
        Y_train_onehot = jnp.eye(4)[Y_train]

        num_batches = int(np.ceil(len(X_train) / batch_size))
        params = random.normal(subkey, (num_params,))
        optimizer = optax.adam(learning_rate=lr)
        optimizer_state = optimizer.init(params)

        loss_curve = []
        for epoch in range(num_epochs):
            key, subkey = random.split(key)
            shuffle_indices = random.permutation(subkey, len(X_train))
            X_shuffled = X_train[shuffle_indices]
            Y_shuffled = Y_train_onehot[shuffle_indices]
            
            batch_losses = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                loss, grads = value_and_grad(loss_fun)(params, X_batch, Y_batch)
                batch_losses.append(loss)
                
                updates, optimizer_state = optimizer.update(grads, optimizer_state)
                params = optax.apply_updates(params, updates)
            
            epoch_loss = np.mean(batch_losses)
            loss_curve.append(epoch_loss)

        return params, loss_curve

    # Train with noise
    trained_params, loss_curve = train_qcnn_local(use_noisy=True)
    
    return {
        'noise': noise_value,
        'trained_params': trained_params,
        'loss_curve': loss_curve
    }

if __name__ == "__main__":
    # Define noise values to train over
    noise_values = [None] + list(np.arange(0.02, 0.22, 0.02))  # None, 0.02, 0.04, ..., 0.20
    
    print(f"Training over {len(noise_values)} noise levels: {noise_values}")
    
    # Use partial to create worker function with fixed parameters
    worker = partial(train_qcnn_for_noise, num_epochs=100, lr=5e-2, T=.5, seed=seed, batch_size=16)
    
    # Multiprocess training
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} cores for parallel training\n")
    
    results_list = []
    with mp.Pool(processes=num_cores) as pool:
        from tqdm import tqdm
        pbar = tqdm(total=len(noise_values), desc="Training over noise levels", unit="model")
        
        for result in pool.imap_unordered(worker, noise_values):
            results_list.append(result)
            pbar.update(1)
        pbar.close()
    
    # Sort results by noise value for convenience
    results_list.sort(key=lambda x: (x['noise'] is None, x['noise']))
    
    print("\nTraining complete!")
    print(f"Trained {len(results_list)} models")
    
    # Extract the best model (noise=None) for predictions
    best_result = next(r for r in results_list if r['noise'] is None)
    trained_params = best_result['trained_params']
    
    # Plot loss curves for comparison
    plt.figure(figsize=(12, 6))
    for result in results_list:
        noise_label = "No noise" if result['noise'] is None else f"Noise={result['noise']:.2f}"
        plt.plot(result['loss_curve'], label=noise_label, alpha=0.7)
    plt.xlabel("Epochs"), plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss Curves for Different Noise Levels")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()


# Take the predicted classes for each point in the phase diagram
predicted_classes = np.argmax(
    vectorized_qcnn_noisy(trained_params, psis.reshape(-1, 2**num_qubits)),
    axis=1
)

colors = ['#80bfff', '#fff2a8',  '#80f090', '#da8080',]
phase_labels = ["Ferromagnetic", "Antiphase", "Paramagnetic", "Trash Class",]
cmap = ListedColormap(colors)

bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

# Plot the predictions over the phase diagram
plt.figure(figsize=(4,4), constrained_layout=True)
plt.imshow(
    predicted_classes.reshape(side, side),
    cmap=cmap,
    norm=norm,
    aspect="auto",
    origin="lower",
    extent=[0, 1, 0, 2]
)

# Plot the transition lines (Ising and KT) for reference.
k_vals1 = np.linspace(0.0, 0.5, 50)
k_vals2 = np.linspace(0.5, 1.0, 50)
plt.plot(k_vals1, ising_transition(k_vals1), 'k')
plt.plot(k_vals2, kt_transition(k_vals2), 'k')
plt.plot(k_vals2, bkt_transition(k_vals2), 'k', ls = '--')

for color, phase in zip(colors, phase_labels[:-1]):
    plt.scatter([], [], color=color, label=phase, edgecolors='black')
plt.plot([], [], 'k', label='Transition lines')

plt.xlabel("k"), plt.ylabel("h")
plt.title("Figure 5. QCNN Classification")
plt.legend()
plt.show()
