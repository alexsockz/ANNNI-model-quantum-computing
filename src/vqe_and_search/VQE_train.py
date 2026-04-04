import pennylane as qml
import numpy as np
from jax import config

import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from VQE import VQE  # Assicurati che la classe VQE sia definita correttamente in VQE.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

config.update("jax_enable_x64", True)

seed = 123456

# Setting our constants
num_qubits = 6         # Numero di qubit
side = 20               # Discretizzazione del diagramma di fase

def compute_point(args):
    """
    Funzione worker che esegue il calcolo per un singolo punto (k, h).
    Riceve una tupla per compatibilità con pool.map.
    """
    x, k_val, y, h_val = args
    
    # Calcolo della fase teorica
    phase = get_phase(k_val, h_val)
    
    # Esecuzione del VQE
    # Nota: il device viene creato all'interno del processo worker
    state, energy_history = get_vqe_state(k_val, h_val, n_layers=12, epochs=4000)
    
    # Restituiamo i risultati insieme agli indici per ricostruire la matrice
    return (y, x, state, phase, energy_history)


def get_H(num_spins, k, h):
    """Costruisce l'Hamiltoniana ANNNI (J=1)"""
    # Interazioni tra primi vicini
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H - (qml.PauliX(i) @ qml.PauliX(i + 1))
    # Interazioni tra secondi vicini
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))
    # Termine di campo magnetico
    for i in range(0, num_spins):
        H = H - h * qml.PauliZ(i)
    return H

def kt_transition(k):
    """Linea di transizione di Kosterlitz-Thouless"""
    return 1.05 * np.sqrt((k - 0.5) * (k - 0.1))

def ising_transition(k):
    """Linea di transizione di Ising"""
    return np.where(k == 0, 1,
                    (1 - k) * (1 - np.sqrt((1 - 3 * k + 4 * k**2) / (1 - k))) / np.maximum(k, 1e-9))

def bkt_transition(k):
    """Linea di transizione della fase flottante"""
    return 1.05 * (k - 0.5)

def get_phase(k, h):
    """Determina la fase teorica in base alle linee DMRG"""
    if k < 0.5 and h < ising_transition(k):
        return 0          # Ferromagnetica
    elif k > 0.5 and h < kt_transition(k):
        return 1          # Antiphase
    return 2              # Paramagnetica

# Crea la meshgrid dei parametri
ks = np.linspace(0, 1, side)
hs = np.linspace(0, 2, side)
K, H_mesh = np.meshgrid(ks, hs)   # H_mesh per non confondere con l'Hamiltoniana

# Prepara l'array per gli stati (complessi)
psis = np.empty((len(ks), len(hs), 2**num_qubits), dtype=np.complex128)

def get_theoretical_energy(k, h):
    """
    Computes the theoretical ground state energy of the ANNNI Hamiltonian.
    """
    H = get_H(num_qubits, k, h)
    dev = qml.device("lightning.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def compute_energy(state):
        for i, amplitude in enumerate(state):
            if amplitude != 0:
                # Construct computational basis state
                binary = format(i, f'0{num_qubits}b')
                for j, bit in enumerate(binary):
                    if bit == '1':
                        qml.PauliX(j)
        return qml.expval(H)
    
    # Find ground state by diagonalization
    H_matrix = qml.matrix(H)
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    return eigenvalues[0]  # Return lowest eigenvalue

def get_vqe_state(k, h, n_layers=9, epochs=1000, max_retries=3, error_threshold=1):
    """
    Esegue il VQE per l'Hamiltoniana con parametri (k, h) e restituisce lo stato fondamentale approssimato.
    Retry se l'errore di run (abs((energyVqe-energyTheoretical)/energyTheoretical)*100) > error_threshold %.
    """
    # Ottieni l'energia teorica
    energy_theoretical = get_theoretical_energy(k, h)
    
    saved_energy=np.inf
    best_state=None
    best_energy_energy_history=None
    for attempt in range(max_retries):
        # Costruisce l'Hamiltoniana
        H = get_H(num_qubits, k, h)
        # Crea il dispositivo (senza rumore per ora)
        dev = qml.device("lightning.qubit", wires=num_qubits)
        # Inizializza la classe VQE (assumendo che accetti k, h e n_layers)
        vqe = VQE(num_qubits, n_layers=n_layers, k=k, h=h, shots=None)
        # Addestra il VQE
        best_energy, _, _, energy_history, _ = vqe.train_VQE(epochs=epochs, non_zero_state=False)
              
        # Calculate run error
        run_error = abs((best_energy - energy_theoretical) / energy_theoretical) * 100
        
        # QNode per estrarre lo stato finale
        @qml.qnode(dev)  # interfaccia di default (NumPy)
        def state_circuit(params):
            vqe.ansatz(params)
            return qml.state()

        params_np = vqe.parameters_vqe.detach().numpy()
        state = state_circuit(params_np)  # state sarà un array NumPy
        
        # Check if error is within threshold
        if run_error <= error_threshold:
            return state, np.array(energy_history)  # già array NumPy, nessuna conversione ulteriore
        else:
            if run_error<saved_energy:
                saved_energy=run_error
                best_state=state
                best_energy_energy_history=np.array(energy_history)
            # Retry if error exceeds threshold and retries are available
            if attempt < max_retries - 1:
                print(f"Run error {run_error:.4f}% > {error_threshold}% threshold for k={k:.3f}, h={h:.3f}. Retrying (attempt {attempt+2}/{max_retries})...")
            else:
                print(f"Max retries reached for k={k:.3f}, h={h:.3f}. Using best result with error {run_error:.4f}%.")
                return best_state, best_energy_energy_history

if __name__ == "__main__":
    # Configurazione JAX (importante farlo dentro il main o prima di definire i worker)
    config.update("jax_enable_x64", True)

    # 1. Preparazione della meshgrid e dei parametri
    ks = np.linspace(0, 1, side)
    hs = np.linspace(0, 2, side)
    
    # Creiamo una lista di tutti i parametri da calcolare
    # Ogni elemento è: (indice_x, valore_k, indice_y, valore_h)
    tasks = []
    for x, k_val in enumerate(ks):
        for y, h_val in enumerate(hs):
            tasks.append((x, k_val, y, h_val))

    print(f"Inizio calcolo parallelo su {len(tasks)} punti...")

    # 2. Avvio del Pool di processi
    # Di default usa tutti i core disponibili. Puoi limitarli con processes=n
    num_cores = mp.cpu_count()
    with mp.Pool(processes=num_cores) as pool:
        # Usiamo imap_unordered con tqdm per la barra di progresso
        pbar = tqdm(total=len(tasks), desc="Computing VQE states", unit="point")
        results = []
        for result in pool.imap_unordered(compute_point, tasks):
            results.append(result)
            pbar.update(1)
        pbar.close()

    # 3. Ricostruzione delle matrici dei risultati
    psis = np.empty((len(hs), len(ks), 2**num_qubits), dtype=np.complex128)
    phases = np.empty((len(hs), len(ks)), dtype=int)
    energy_histories = np.empty((len(hs), len(ks)), dtype=object)

    for y, x, state, phase, energy_history in results:
        psis[y, x] = state
        phases[y, x] = phase
        energy_histories[y, x] = energy_history

    # 4. Salvataggio
    np.savez(PROJECT_ROOT / "vqe_states.npz", psis=psis, ks=ks, hs=hs, phases=phases, energy_histories=energy_histories)

    print("Calcolo completato e stati salvati.")