import pennylane as qml
import numpy as np
from jax import jit, vmap, value_and_grad, random, config
from jax import numpy as jnp
import optax
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

seed = 123465
num_qubits = 8
side = 20

# ------------------------------
# Definizione delle funzioni principali (non modificate)
# ------------------------------
def get_H(num_spins, k, h):
    """Costruzione dell'Hamiltoniana ANNNI (J=1)"""
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H - (qml.PauliX(i) @ qml.PauliX(i + 1))
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))
    for i in range(0, num_spins):
        H = H - h * qml.PauliZ(i)
    return H

def kt_transition(k):
    return 1.05 * np.sqrt((k - 0.5) * (k - 0.1))

def ising_transition(k):
    return np.where(k == 0, 1, (1 - k) * (1 - np.sqrt((1 - 3 * k + 4 * k**2) / (1 - k))) / np.maximum(k, 1e-9))

def get_phase(k, h):
    if k < 0.5 and h < ising_transition(k):
        return 0
    elif k > 0.5 and h < kt_transition(k):
        return 1
    return 2

def diagonalize_H(H_matrix):
    _, psi = jnp.linalg.eigh(H_matrix)
    return jnp.array(psi[:, 0], dtype=jnp.complex64)

# Preparazione dei dati di input (psi per tutti i punti del diagramma)
ks = np.linspace(0, 1, side)
hs = np.linspace(0, 2, side)
K, H = np.meshgrid(ks, hs)

H_matrices = np.empty((len(ks), len(hs), 2**num_qubits, 2**num_qubits))
phases = np.empty((len(ks), len(hs)), dtype=int)

for x, k in enumerate(ks):
    for y, h in enumerate(hs):
        H_matrices[y, x] = np.real(qml.matrix(get_H(num_qubits, k, h)))
        phases[y, x] = get_phase(k, h)

psis = vmap(vmap(diagonalize_H))(H_matrices)  # shape (side, side, 2**num_qubits)

# ------------------------------
# Definizione degli ansatz (con/senza rumore)
# ------------------------------
def anomaly_ansatz(n_qubit, params, noise_strength=None):
    """
    Ansatz per il circuito QAD.
    Se noise_strength è None, non vengono inseriti canali di depolarizzazione.
    """
    def block(nontrash, trash, shift):
        # Collegamenti tra trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
            if noise_strength is not None:
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
                qml.DepolarizingChannel(noise_strength, wires=int(target))
        # Collegamenti nontrash -> trash
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])
            if noise_strength is not None:
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
                qml.DepolarizingChannel(noise_strength, wires=int(trash[trash_idx]))

    depth = 2
    n_trashwire = n_qubit // 2
    trash = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)
    nontrash = np.setdiff1d(np.arange(n_qubit), trash)

    index = 0
    # Layer iniziale
    for wire in np.arange(n_qubit):
        qml.RY(params[index], wires=int(wire))
        if noise_strength is not None:
            qml.DepolarizingChannel(noise_strength, wires=int(wire))
        index += 1

    # Blocchi ripetuti
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        wires_to_rot = np.arange(n_qubit) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            index += 1

    return index, list(trash)

def build_qnode(noise_strength):
    """
    Crea il QNode appropriato in base alla presenza o meno di rumore.
    """
    device_name = "default.mixed" if noise_strength is not None else "default.qubit"
    dev = qml.device(device_name, wires=num_qubits)

    @qml.qnode(dev)
    def circuit(params, state):
        qml.StatePrep(state, wires=range(num_qubits), normalize=True)
        _, trash_wires = anomaly_ansatz(num_qubits, params, noise_strength)
        return [qml.expval(qml.PauliZ(int(k))) for k in trash_wires]

    return circuit

def train_anomaly(noise_strength, num_epochs=100, lr=1e-1, seed=123456):
    """
    Allena il modello QAD per un dato livello di rumore.
    Restituisce (parametri, lista dei loss, QNode compilato).
    """
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    # Costruisci il QNode e le versioni compilate/vettorizzate
    qnode = build_qnode(noise_strength)
    jitted_qnode = jit(qnode)
    vectorized_qnode = vmap(jitted_qnode, in_axes=(None, 0))

    # Training set: solo lo stato con k=0, h=0
    X_train = jnp.array(psis[0, 0])

    # Inizializzazione casuale dei parametri
    params = random.normal(subkey, (num_anomaly_params,))

    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    loss_curve = []

    def loss_fun(params, X):
        score = 1 - jnp.array(jitted_qnode(params, X))
        return jnp.mean(score)

    for epoch in range(num_epochs):
        loss, grads = value_and_grad(loss_fun)(params, X_train)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        loss_curve.append(loss)

    return params, loss_curve, vectorized_qnode

# ------------------------------
# Esecuzione per diversi livelli di rumore
# ------------------------------
# Calcoliamo il numero di parametri una volta sola (non dipende dal rumore)
_, trash_wires_temp = anomaly_ansatz(num_qubits, [0]*100, None)
num_anomaly_params, _ = anomaly_ansatz(num_qubits, [0]*100, None)

# Lista dei livelli di rumore da testare (None = senza rumore)
noise_levels = [None, 0.05, 0.10, 0.15]
loss_curves = {}
trained_params = {}

for noise in noise_levels:
    print(f"\nTraining con noise_strength = {noise}")
    params, loss_curve, _ = train_anomaly(noise, num_epochs=100, lr=1e-1, seed=seed)
    loss_curves[noise] = loss_curve
    trained_params[noise] = params

# ------------------------------
# Plot delle curve di loss
# ------------------------------
plt.figure(figsize=(8,5))
for noise, loss_curve in loss_curves.items():
    label = f"Noise = {noise}" if noise is not None else "No noise"
    plt.plot(loss_curve, label=label, linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Compression Loss")
plt.title("Training loss curves for different noise levels")
plt.legend()
plt.grid()
plt.savefig("noise_train.png")