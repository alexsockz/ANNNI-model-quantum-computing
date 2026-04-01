import pennylane as qml
import numpy as np
from jax import jit, vmap, value_and_grad, random, config
from jax import numpy as jnp
import optax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

config.update("jax_enable_x64", True)

seed = 123456

# Setting our constants
num_qubits = 6 # Number of spins in the Hamiltonian (= number of qubits)
side = 20      # Discretization of the Phase Diagram

answer = input("noise y or n?").lower().strip()
if answer == "y":
    noise_strength = 0.75
elif answer == "n":
    noise_strength = None


try:
    npzfile = np.load("../../../vqe_states.npz", allow_pickle=True)
    print("File loaded. Keys:", npzfile.files)
except FileNotFoundError:
    print("File 'vqe_results.db' non trovato. Assicurati che esista nella cartella corrente.")
    exit()

ks = npzfile["ks"]          # array dei valori di k (dimensione side)
hs = npzfile["hs"]          # array dei valori di h (dimensione side)
psis = npzfile["psis"]      # array degli stati: shape (len(hs), len(ks), 2**num_qubits)


# Verifica dimensioni
print(f"ks shape: {ks.shape}, hs shape: {hs.shape}, psis shape: {psis.shape}")


psis = jnp.array(psis, dtype=jnp.complex64)

K, H = np.meshgrid(ks, hs)


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

def anomaly_noisy(n_qubit, params):
    """Ansatz of the QAD model
    Apply multi-qubit gates between trash and non-trash wires
    """

    # Block of gates connecting trash and non-trash wires
    def block(nontrash, trash, shift):
        # Connect trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
                qml.DepolarizingChannel(noise_strength, wires=int(target))
        # Connect each nontrash wire to a trash wire
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
                qml.DepolarizingChannel(noise_strength, wires=int(trash[trash_idx]))
    depth = 2  # Number of repeated block layers
    n_trashwire = n_qubit // 2

    # Define trash wires as a contiguous block in the middle.
    trash = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)
    nontrash = np.setdiff1d(np.arange(n_qubit), trash)

    index = 0

    # Initial layer: apply RY rotations on all wires.
    for wire in np.arange(n_qubit):
        qml.RY(params[index], wires=int(wire))
        if answer == "y":
            qml.DepolarizingChannel(noise_strength, wires=int(wire))

        index += 1

    # Repeatedly apply blocks of entangling gates and additional rotations.
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        # In the final layer, only apply rotations on trash wires.
        wires_to_rot = np.arange(n_qubit) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
            index += 1
    return index, list(trash)


@qml.qnode(qml.device("default.mixed", wires=num_qubits))
def anomalynode_noisy(params, state):
    """QNode with QAD ansatz and expectation values of the trash wires as output"""
    # Input ground state from diagonalization
    qml.StatePrep(state, wires=range(num_qubits), normalize = True)
    # Quantum Anomaly Circuit
    _, trash_wires = anomaly_noisy(num_qubits, params)


    return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]



# Vectorize the circuit using vmap
jitted_anomaly_circuit = jit(anomaly_circuit)
vectorized_anomaly_circuit = vmap(jitted_anomaly_circuit, in_axes=(None, 0))

jitted_anomalynode_noisy = jit(anomalynode_noisy)
vectorized_anomalynode_noisy = vmap(jitted_anomalynode_noisy, in_axes=(None, 0))

# Draw the QAD Architecture
fig,ax = qml.draw_mpl(anomaly_circuit)(np.arange(num_anomaly_params), psis[0,0])
fig.savefig("circuito_anomaly.png", dpi=300, bbox_inches="tight")

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
plt.show()

# Evaluate the compression score for each state in the phase diagram
if answer == "y":
    compressions = vectorized_anomalynode_noisy(trained_anomaly_params, psis.reshape(-1, 2**num_qubits))
elif answer == "n":
    compressions = vectorized_anomaly_circuit(trained_anomaly_params, psis.reshape(-1, 2 ** num_qubits))
compressions = jnp.mean(1 - jnp.array(compressions), axis = 0)

im = plt.imshow(compressions.reshape(len(hs), len(ks)),
                aspect="auto",
                origin="lower",
                extent=[ks.min(), ks.max(), hs.min(), hs.max()],
                cmap='viridis')  # Puoi scegliere una colormap

# Linee di transizione (definite sugli stessi intervalli di k)
plt.plot(np.linspace(0.0, 0.5, 50), ising_transition(np.linspace(0.0, 0.5, 50)), 'k')
plt.plot(np.linspace(0.5, 1.0, 50), kt_transition(np.linspace(0.5, 1.0, 50)), 'k')

# Punto di addestramento (k=ks[0], h=hs[0])
plt.scatter([ks[0]], [hs[0]], color='r', marker='x', label="Training point", s=50)

plt.legend()
plt.xlabel("k")
plt.ylabel("h")
plt.title("Figure 7. Phase diagram with QAD")
cbar = plt.colorbar(im)
cbar.set_label(r"Compression Score  $\mathcal{C}$")
plt.show()