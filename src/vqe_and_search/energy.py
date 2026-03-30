import pennylane as qml
import numpy as npa
from jax import jit, vmap, value_and_grad, random, config
from jax import numpy as jnp
import optax

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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

def theoretical_energy(N,k,h):
    H_annni = get_H(N, k, h)

    @qml.qnode(qml.device("default.qubit", wires=N))
    def get_energy(state_vector):
        qml.StatePrep(state_vector, wires=range(N), normalize = True)
        return qml.expval(H_annni)


    H_matrices = npa.empty((1, 1, 2**N, 2**N))
        
    # Fill H_matrices with actual Hamiltonian matrices

    H_matrices[0,0] = npa.real(qml.matrix(get_H(N, k, h)))

    def diagonalize_H(H_matrix):
        """Returns the lowest eigenvector of the Hamiltonian matrix."""
        _, psi = jnp.linalg.eigh(H_matrix)  # Compute eigenvalues and eigenvectors
        return jnp.array(psi[:, 0], dtype=jnp.complex64)  # Return the ground state

    # Vectorized diagonalization
    psis = vmap(vmap(diagonalize_H))(H_matrices)
    energy = get_energy(psis[0,0])
    return energy

if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    seed = 123456

    # Setting our constants
    num_qubits = 8 # Number of spins in the Hamiltonian (= number of qubits)
    side = 20      # Discretization of the Phase Diagram

    k = 0.5
    h = 1.0
    from src.vqe_and_search.VQE import VQE
    from time import perf_counter
    print("begin training")
    t1=perf_counter()
    vqe=VQE(num_qubits,8,k,h)
    best_energy=vqe.train_VQE()
    print(perf_counter()-t1)
    print(theoretical_energy(num_qubits,k,h))
    print(best_energy[0])