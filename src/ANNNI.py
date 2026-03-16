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
side = 20      # Discretization of the Phase Diagram

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
    #eigenvalue, eigenvectors=
    _, psi = jnp.linalg.eigh(H_matrix)  # Compute eigenvalues and eigenvectors
    return jnp.array(psi[:, 0], dtype=jnp.complex64)  # Return the ground state


# Create meshgrid of the parameter space
ks = [0.2]
hs = [0.5]
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

print(len(psis[0,0]))
# Calculate the energy of the system using the ground state psi and Hamiltonian
energy = jnp.real(jnp.dot(psis[0,0].conj().T, jnp.dot(H_matrices[0,0], psis[0,0])))
print("ANNNI energy:", energy)