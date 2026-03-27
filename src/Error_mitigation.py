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
num_qubits = 8 # Number of spins in the Hamiltonian (= number of qubits)
side = 20     # Discretization of the Phase Diagram

answer = input("noise y or n?").lower().strip()
if answer == "y":
    noise_strength = 0.15
elif answer == "n":
    noise_strength = None


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


def qcnn_ansatz(num_qubits, params):
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
    active_wires = np.arange(num_qubits)
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

num_params, output_wires = qcnn_ansatz(num_qubits, [0]*100)

@qml.qnode(qml.device("default.qubit", wires=num_qubits))
def qcnn_circuit(params, state):
    """QNode with QCNN ansatz and probabilities of unmeasured qubits as output"""
    # Input ground state from diagonalization
    qml.StatePrep(state, wires=range(num_qubits), normalize = True)
    # QCNN
    _, output_wires = qcnn_ansatz(num_qubits, params)

    return qml.probs([int(k) for k in output_wires])

# Vectorized circuit through vmap
vectorized_qcnn_circuit = vmap(jit(qcnn_circuit), in_axes=(None, 0))

# Draw the QCNN Architecture
fig,ax = qml.draw_mpl(qcnn_circuit)(np.arange(num_params), psis[0,0])


def qcnn_ansatz_noisy(num_qubits, params):
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
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wires[-1]))
            index += 1

        for group in groups:
            qml.CNOT(wires=[int(group[0]), int(group[1])])
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(group[0]))
                qml.DepolarizingChannel(noise_strength, wires=int(group[1]))
            for wire in group:
                qml.RY(params[index], wires=int(wire))
                if answer == "y":
                    qml.DepolarizingChannel(noise_strength, wires=int(wire))
                index += 1
        return index

    # Pooiling block
    def pool(wires, params, index):
        for wire_pool, wire in zip(wires[0::2], wires[1::2]):
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wire_pool))
            m_0 = qml.measure(int(wire_pool))
            # Nota: dopo una misura non mettiamo rumore perché lo stato è collassato
            qml.cond(m_0 == 0, qml.RX)(params[index], wires=int(wire))
            qml.cond(m_0 == 1, qml.RX)(params[index + 1], wires=int(wire))
            # Dopo le RX condizionali, aggiungiamo rumore sui qubit ancora attivi
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wire))
            index += 2
            wires = np.delete(wires, np.where(wires == wire_pool))

        if len(wires) % 2 != 0:
            qml.RX(params[index], wires=int(wires[-1]))
            if answer == "y":
                qml.DepolarizingChannel(noise_strength, wires=int(wires[-1]))
            index += 1
        return index, wires

    # Initialize active wires and parameter index.
    active_wires = np.arange(num_qubits)
    index = 0

    # Initial layer: apply RY to all wires.
    for wire in active_wires:
        qml.RY(params[index], wires=int(wire))
        if answer == "y":
            qml.DepolarizingChannel(noise_strength, wires=int(wire))
        index += 1

    # Repeatedly apply convolution and pooling until there are 2 unmeasured wires
    while len(active_wires) > 2:
        index = conv(active_wires, params, index)
        index, active_wires = pool(active_wires, params, index)
        qml.Barrier()

    # Final layer: apply RY to the remaining active wires.
    for wire in active_wires:
        qml.RY(params[index], wires=int(wire))
        if answer == "y":
            qml.DepolarizingChannel(noise_strength, wires=int(wire))
        index += 1
    return index, active_wires

num_params, output_wires = qcnn_ansatz_noisy(num_qubits, [0]*100)


if answer == "y":
    dev = qml.device("default.mixed", wires=num_qubits)
elif answer == "n":
    dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def qcnn_noisy(params, state):
    qml.StatePrep(state, wires=range(num_qubits), normalize=True)
    _, output_wires = qcnn_ansatz_noisy(num_qubits, params)
    if answer == "y":
        qml.DepolarizingChannel(noise_strength, wires=[int(k) for k in output_wires])
    return qml.probs([int(k) for k in output_wires])

# Vectorized circuit through vmap
vectorized_qcnn_noisy = vmap(jit(qcnn_noisy), in_axes=(None, 0))


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

def train_qcnn(num_epochs, lr, T, seed):
    """Training function of the QCNN architecture"""

    # Initialize PRNG key
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)

    # Define the loss function
    def loss_fun(params, X, Y):
        preds = vectorized_qcnn_circuit(params, X)
        return cross_entropy(preds, Y, T)

    # Consider only analytical points for the training
    X_train, Y_train = psis[analytical_mask], phases[analytical_mask]

    # Convert labels to one-hot encoding
    Y_train_onehot = jnp.eye(4)[Y_train]

    # Randomly initialize the parameters
    params = random.normal(subkey, (num_params,))

    # Initialize Adam optimizer
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)


    loss_curve = []
    for epoch in range(num_epochs):
        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fun)(params, X_train, Y_train_onehot)

        # Update parameters
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        loss_curve.append(loss)

    return params, loss_curve

trained_params, loss_curve = train_qcnn(num_epochs=100, lr=5e-2, T=.5, seed=seed)


# Plot the loss curve
plt.figure()
plt.plot(loss_curve, label="Loss", color="blue", linewidth=2)
plt.xlabel("Epochs"), plt.ylabel("Cross-Entropy Loss")
plt.title("Figure 4. QCNN Training Cross-Entropy Loss Curve")
plt.legend()
plt.grid()
plt.show()



# Take the predicted classes for each point in the phase diagram
probis = vectorized_qcnn_noisy(trained_params, psis.reshape(-1, 2**num_qubits))
predicted_classes = np.argmax(probis, axis=1)

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


if answer == "y":
    def qcnn_ansatz_scaled(num_qubits, params, scale):
        """Restituisce le probabilità per un dato stato e un fattore di scala del rumore."""
        # Crea un device misto con lo stesso numero di qubit
        dev_scaled = qml.device("default.mixed", wires=num_qubits)

        # Convolution block
        def conv(wires, params, index):
            if len(wires) % 2 == 0:
                groups = wires.reshape(-1, 2)
            else:
                groups = wires[:-1].reshape(-1, 2)
                qml.RY(params[index], wires=int(wires[-1]))
                if answer == "y":
                    qml.DepolarizingChannel(noise_strength * scale, wires=int(wires[-1]))
                index += 1

            for group in groups:
                qml.CNOT(wires=[int(group[0]), int(group[1])])
                if answer == "y":
                    qml.DepolarizingChannel(noise_strength * scale, wires=int(group[0]))
                    qml.DepolarizingChannel(noise_strength * scale, wires=int(group[1]))
                for wire in group:
                    qml.RY(params[index], wires=int(wire))
                    if answer == "y":
                        qml.DepolarizingChannel(noise_strength * scale, wires=int(wire))
                    index += 1
            return index

        # Pooiling block
        def pool(wires, params, index):
            for wire_pool, wire in zip(wires[0::2], wires[1::2]):
                if answer == "y":
                    qml.DepolarizingChannel(noise_strength * scale, wires=int(wire_pool))
                m_0 = qml.measure(int(wire_pool))
                # Nota: dopo una misura non mettiamo rumore perché lo stato è collassato
                qml.cond(m_0 == 0, qml.RX)(params[index], wires=int(wire))
                qml.cond(m_0 == 1, qml.RX)(params[index + 1], wires=int(wire))
                # Dopo le RX condizionali, aggiungiamo rumore sui qubit ancora attivi
                if answer == "y":
                    qml.DepolarizingChannel(noise_strength * scale, wires=int(wire))
                index += 2
                wires = np.delete(wires, np.where(wires == wire_pool))

            if len(wires) % 2 != 0:
                qml.RX(params[index], wires=int(wires[-1]))
                if answer == "y":
                    qml.DepolarizingChannel(noise_strength * scale, wires=int(wires[-1]))
                index += 1
            return index, wires

        # Initialize active wires and parameter index.
        active_wires = np.arange(num_qubits)
        index = 0

        # Initial layer: apply RY to all wires.
        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            if answer == "y":
                qml.DepolarizingChannel(noise_strength * scale, wires=int(wire))
            index += 1

        # Repeatedly apply convolution and pooling until there are 2 unmeasured wires
        while len(active_wires) > 2:
            index = conv(active_wires, params, index)
            index, active_wires = pool(active_wires, params, index)
            qml.Barrier()

        # Final layer: apply RY to the remaining active wires.
        for wire in active_wires:
            qml.RY(params[index], wires=int(wire))
            if answer == "y":
                qml.DepolarizingChannel(noise_strength * scale, wires=int(wire))
            index += 1
        return index, active_wires

    def qcnn_with_scaled_noise(params, state, scale):
        """Restituisce le probabilità per un dato stato e un fattore di scala del rumore."""
        # Crea un device misto con lo stesso numero di qubit
        dev_scaled = qml.device("default.mixed", wires=num_qubits)

        @qml.qnode(dev_scaled)
        def circuit(params, state):
            qml.StatePrep(state, wires=range(num_qubits), normalize=True)
            # Usa una versione della ansatz in cui la forza del rumore è moltiplicata per scale
            _, output_wires = qcnn_ansatz_scaled(num_qubits, params, scale)
            if answer == "y":
                qml.DepolarizingChannel(noise_strength * scale, wires=[int(k) for k in output_wires])
            return qml.probs([int(k) for k in output_wires])

        return circuit(params, state)


    def extrapolate_linear_3points(probs_1x, probs_2x, probs_3x, scales):
        """
        Estrapola linearmente a rumore zero usando i tre punti.
        Restituisce probabilità mitigate di forma (N, C).
        """
        p1 = np.asarray(probs_1x)
        p2 = np.asarray(probs_2x)
        p3 = np.asarray(probs_3x)
        N, C = p1.shape
        probs_mit = np.zeros((N, C))

        for i in range(N):
            for j in range(C):
                y = [p1[i, j], p2[i, j], p3[i, j]]
                coeffs = np.polyfit(scales, y, 1)  # coeffs = [pendenza, intercetta]
                probs_mit[i, j] = coeffs[1]  # intercetta a x=0

        # Clipping e normalizzazione per ottenere probabilità valide
        probs_mit = np.clip(probs_mit, 0, 1)
        probs_mit /= probs_mit.sum(axis=1, keepdims=True)
        return probs_mit

    # Vettorizza la funzione che calcola le probabilità per un dato scale
    vmap_qcnn_scaled = vmap(lambda state, scale: qcnn_with_scaled_noise(trained_params, state, scale),
                            in_axes=(0, None))

    # Lista dei fattori di scala da usare
    scales = [1.0 , 1.5, 2.0]

    # Calcola le probabilità per ogni scala
    probs_by_scale = []
    for s in scales:
        if s == 1.0:
            probs = probis
        else:
            probs = vmap_qcnn_scaled(psis.reshape(-1, 2**num_qubits), s)
        probs_by_scale.append(probs)


    probs_mitigated = extrapolate_linear_3points(probs_by_scale[0], probs_by_scale[1], probs_by_scale[2], scales)
    predicted_mitclasses = np.argmax(probs_mitigated, axis=1)




    # Take the predicted classes for each point in the phase diagram

    colors = ['#80bfff', '#fff2a8',  '#80f090', '#da8080',]
    phase_labels = ["Ferromagnetic", "Antiphase", "Paramagnetic", "Trash Class",]
    cmap = ListedColormap(colors)

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot the predictions over the phase diagram
    plt.figure(figsize=(4,4), constrained_layout=True)
    plt.imshow(
        predicted_mitclasses.reshape(side, side),
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
    plt.title("Figure 6. QCNN Classification mitigated")
    plt.legend()
    plt.show()