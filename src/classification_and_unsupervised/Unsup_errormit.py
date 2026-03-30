import jax
import pennylane as qml
import numpy as np
from jax import jit, vmap, value_and_grad, random, config
from jax import numpy as jnp
import optax
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

config.update("jax_enable_x64", True)

seed = 123456

# Setting our constants
num_qubits = 8 # Number of spins in the Hamiltonian (= number of qubits)
side = 20      # Discretization of the Phase Diagram

os.makedirs("plots_unsup_errormit", exist_ok=True)


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


def anomaly_ansatz(n_qubit, params):
    """Ansatz of the QAD model
    Apply multi-qubit gates between trash and non-trash wires
    """

    # Block of gates connecting trash and non-trash wires
    def block(nontrash, trash, shift):
        # Connect trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
        # Connect each nontrash wire to a trash wire
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])

    depth = 2  # Number of repeated block layers
    n_trashwire = n_qubit // 2

    # Define trash wires as a contiguous block in the middle.
    trash = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)
    nontrash = np.setdiff1d(np.arange(n_qubit), trash)

    index = 0

    # Initial layer: apply RY rotations on all wires.
    for wire in np.arange(n_qubit):
        qml.RY(params[index], wires=int(wire))
        index += 1

    # Repeatedly apply blocks of entangling gates and additional rotations.
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        # In the final layer, only apply rotations on trash wires.
        wires_to_rot = np.arange(n_qubit) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            index += 1

    return index, list(trash)

num_anomaly_params, trash_wires = anomaly_ansatz(num_qubits, [0]*100)

@qml.qnode(qml.device("default.qubit", wires=num_qubits))
def anomaly_circuit(params, state):
    """QNode with QAD ansatz and expectation values of the trash wires as output"""
    # Input ground state from diagonalization
    qml.StatePrep(state, wires=range(num_qubits), normalize = True)
    # Quantum Anomaly Circuit
    _, trash_wires = anomaly_ansatz(num_qubits, params)

    return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]

def anomaly_noisy(n_qubit, params, current_noise_strength):
    """Ansatz of the QAD model
    Apply multi-qubit gates between trash and non-trash wires
    """

    # Block of gates connecting trash and non-trash wires
    def block(nontrash, trash, shift):
        # Connect trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
            if current_noise_strength > 0:
                qml.DepolarizingChannel(current_noise_strength, wires=int(wire))
                qml.DepolarizingChannel(current_noise_strength, wires=int(target))
        # Connect each nontrash wire to a trash wire
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])
            if current_noise_strength > 0:
                qml.DepolarizingChannel(current_noise_strength, wires=int(wire))
                qml.DepolarizingChannel(current_noise_strength, wires=int(trash[trash_idx]))
    depth = 2  # Number of repeated block layers
    n_trashwire = n_qubit // 2

    # Define trash wires as a contiguous block in the middle.
    trash = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)
    nontrash = np.setdiff1d(np.arange(n_qubit), trash)

    index = 0

    # Initial layer: apply RY rotations on all wires.
    for wire in np.arange(n_qubit):
        qml.RY(params[index], wires=int(wire))
        if current_noise_strength > 0:
            qml.DepolarizingChannel(current_noise_strength, wires=int(wire))

        index += 1

    # Repeatedly apply blocks of entangling gates and additional rotations.
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        # In the final layer, only apply rotations on trash wires.
        wires_to_rot = np.arange(n_qubit) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            if current_noise_strength > 0:
                qml.DepolarizingChannel(current_noise_strength, wires=int(wire))

            index += 1

    return index, list(trash)


@qml.qnode(qml.device("default.qubit", wires=num_qubits))
def anomaly_circuit(params, state):
    """QNode with QAD ansatz and expectation values of the trash wires as output"""
    # Input ground state from diagonalization
    qml.StatePrep(state, wires=range(num_qubits), normalize = True)
    # Quantum Anomaly Circuit
    _, trash_wires = anomaly_ansatz(num_qubits, params)

    return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]


# Vectorize the circuit using vmap
jitted_anomaly_circuit = jit(anomaly_circuit)
vectorized_anomaly_circuit = vmap(jitted_anomaly_circuit, in_axes=(None, 0))

# Draw the QAD Architecture
fig,ax = qml.draw_mpl(anomaly_circuit)(np.arange(num_anomaly_params), psis[0,0])
plt.close(fig)

def train_anomaly(num_epochs, lr, seed):
    """Training function of the QAD architecture"""

    # Initialize PRNG key
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    # Define the loss function
    def loss_fun(params, X):
        # Output expectation values of the qubits
        score = 1 - jnp.array(jitted_anomaly_circuit(params, X))
        loss_value = jnp.mean(score)

        return loss_value

    # Training set consists only of the k = 0 and h = 0 state
    X_train = jnp.array(psis[0, 0])

    # Randomly initialize parameters
    params = random.normal(subkey, (num_anomaly_params,))

    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    loss_curve = []
    for epoch in range(num_epochs):
        # Get random indices for a batch
        key, subkey = random.split(key)

        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fun)(params, X_train)

        # Update parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        loss_curve.append(loss)

    return params, loss_curve

trained_anomaly_params, anomaly_loss_curve = train_anomaly(num_epochs=100, lr=1e-1, seed=seed)

# Plot the loss curve
plt.figure()
plt.plot(anomaly_loss_curve, label="Loss", color="blue", linewidth=2)
plt.xlabel("Epochs"), plt.ylabel("Compression Loss")
plt.title("Figure 6. Anomaly training compression loss curve")
plt.legend()
plt.grid()
plt.savefig("plots_unsup_errormit/anomaly_loss_curve.png")
plt.close()

# Generate phase diagrams over multiple noise levels (0% to 100% in 5% steps) with error mitigation
def anomaly_noisy_scaled(n_qubit, params, scale, current_noise_strength):
    """Ansatz of the QAD model with scaled noise for error mitigation."""

    # Block of gates connecting trash and non-trash wires
    def block(nontrash, trash, shift):
        # Connect trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
            noise_param = min(current_noise_strength * scale, 1.0)
            if noise_param > 0:
                qml.DepolarizingChannel(noise_param, wires=int(wire))
                qml.DepolarizingChannel(noise_param, wires=int(target))
        # Connect each nontrash wire to a trash wire
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])
            noise_param = min(current_noise_strength * scale, 1.0)
            if noise_param > 0:
                qml.DepolarizingChannel(noise_param, wires=int(wire))
                qml.DepolarizingChannel(noise_param, wires=int(trash[trash_idx]))

    depth = 2  # Number of repeated block layers
    n_trashwire = n_qubit // 2

    # Define trash wires as a contiguous block in the middle.
    trash = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)
    nontrash = np.setdiff1d(np.arange(n_qubit), trash)

    index = 0

    # Initial layer: apply RY rotations on all wires.
    for wire in np.arange(n_qubit):
        qml.RY(params[index], wires=int(wire))
        noise_param = min(current_noise_strength * scale, 1.0)
        if noise_param > 0:
            qml.DepolarizingChannel(noise_param, wires=int(wire))

        index += 1

    # Repeatedly apply blocks of entangling gates and additional rotations.
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        # In the final layer, only apply rotations on trash wires.
        wires_to_rot = np.arange(n_qubit) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            noise_param = min(current_noise_strength * scale, 1.0)
            if noise_param > 0:
                qml.DepolarizingChannel(noise_param, wires=int(wire))

            index += 1

    return index, list(trash)


def extrapolate_points(exps_list, scales):
    """Extrapolate linearly to zero noise using polynomial fit."""
    n_scales = len(exps_list)
    exps_stack = np.array(exps_list)
    N, n_trash = exps_stack.shape[1], exps_stack.shape[2]

    exp_mit = np.zeros((N, n_trash))

    for i in range(N):
        for j in range(n_trash):
            y = exps_stack[:, i, j]
            coeffs = np.polyfit(scales, y, deg=n_scales - 1)
            exp_mit[i, j] = coeffs[-1]

    return exp_mit


noise_levels = np.arange(0.0, 1.05, 0.05)

for ns in noise_levels:
    print(f"Evaluating and plotting phase diagram for noise strength: {ns:.2f}")

    @qml.qnode(qml.device("default.mixed", wires=num_qubits))
    def anomalynode_noisy_eval(params, state):
        qml.StatePrep(state, wires=range(num_qubits), normalize=True)
        _, trash_wires = anomaly_noisy(num_qubits, params, ns)
        return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]

    jitted_eval = jit(anomalynode_noisy_eval)
    vectorized_eval = vmap(jitted_eval, in_axes=(None, 0))

    # Evaluate the compression score for each state in the phase diagram
    compressions = vectorized_eval(trained_anomaly_params, psis.reshape(-1, 2**num_qubits))
    compressions = jnp.mean(1 - jnp.array(compressions), axis=0)

    plt.figure()
    im = plt.imshow(compressions.reshape(side, side), aspect="auto", origin="lower", extent=[0, 1, 0, 2])

    # Plot transition lines
    plt.plot(np.linspace(0.0, 0.5, 50), ising_transition(np.linspace(0.0, 0.5, 50)), 'k')
    plt.plot(np.linspace(0.5, 1.0, 50), kt_transition(np.linspace(0.5, 1.0, 50)), 'k')

    plt.plot([], [], 'k', label='Transition Lines')
    plt.scatter([0 +.3/len(ks)], [0 + .5/len(hs)], color='r', marker='x', label="Training point", s=50)

    plt.legend()
    plt.xlabel("k")
    plt.ylabel("h")
    plt.title(f"Figure 7. Phase diagram with QAD (Noise: {int(np.round(ns*100))}%)")
    cbar = plt.colorbar(im)
    cbar.set_label(r"Compression Score  $\mathcal{C}$")

    plt.savefig(f"plots_unsup_errormit/phase_diagram_noise_{int(np.round(ns*100)):03d}.png")
    plt.close()

    # Error mitigation with scaled noise
    def anomaly_with_scaled_noise(params, state, scale):
        dev_scaled = qml.device("default.mixed", wires=num_qubits)

        @qml.qnode(dev_scaled)
        def circuit(params, state):
            qml.StatePrep(state, wires=range(num_qubits), normalize=True)
            _, trash_wires = anomaly_noisy_scaled(num_qubits, params, scale, ns)
            return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]

        return circuit(params, state)

    vmap_anomaly_scaled = vmap(lambda state, scale: anomaly_with_scaled_noise(trained_anomaly_params, state, scale),
                               in_axes=(0, None))

    scales = [1.0, 1.5, 2.0]
    exps_by_scale = []
    for s in scales:
        exps = vmap_anomaly_scaled(psis.reshape(-1, 2**num_qubits), s)
        exps_by_scale.append(exps)

    exp_mitigated = extrapolate_points(exps_by_scale, scales)
    compressions_mit = jnp.mean(1 - jnp.array(exp_mitigated), axis=0)

    plt.figure()
    im = plt.imshow(compressions_mit.reshape(side, side), aspect="auto", origin="lower", extent=[0, 1, 0, 2])

    plt.plot(np.linspace(0.0, 0.5, 50), ising_transition(np.linspace(0.0, 0.5, 50)), 'k')
    plt.plot(np.linspace(0.5, 1.0, 50), kt_transition(np.linspace(0.5, 1.0, 50)), 'k')

    plt.plot([], [], 'k', label='Transition Lines')
    plt.scatter([0 + .3 / len(ks)], [0 + .5 / len(hs)], color='r', marker='x', label="Training point", s=50)

    plt.legend()
    plt.xlabel("k")
    plt.ylabel("h")
    plt.title(f"Figure 8. Mitigated phase diagram with QAD (Noise: {int(np.round(ns*100))}%)")
    cbar = plt.colorbar(im)
    cbar.set_label(r"Compression Score  $\mathcal{C}$")

    plt.savefig(f"plots_unsup_errormit/phase_diagram_mitigated_noise_{int(np.round(ns*100)):03d}.png")
    plt.close()

print("All noise levels processed successfully! Check the 'plots_unsup_errormit' directory.")