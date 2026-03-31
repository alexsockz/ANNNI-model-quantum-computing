import numpy as np
import energy
import pennylane as qml

npzfile = np.load("../../../vqe_states.npz")
print(npzfile.files)

i=0
print(npzfile["psis"][i])
print(npzfile["ks"][i])
print(npzfile["hs"][i])
print(npzfile["phases"][i][i])

N=8

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

dev = qml.device("lightning.qubit", wires=N)
@qml.qnode(dev)
def compute_energy_expval(state,H,N):
    qml.StatePrep(state, wires=range(N), normalize = True)
    return qml.expval(H)

from time import perf_counter


ttot=0
ttott=0
# Check normalization
#sum=0
#for i in range(len(params)):
#    sum += params[i]**2
#dist = np.sqrt(sum)

cose=np.empty((len(npzfile["ks"]),len(npzfile["hs"])))
for i,k in enumerate(npzfile["ks"]):
    for j,h in enumerate(npzfile["hs"]):
        H = get_H(N, k, h)
        t1=perf_counter()
        computed=compute_energy_expval(npzfile["psis"][j,i], H, N)
        ttot+=perf_counter()-t1
        if(j%20==0):print(f"expval {ttot}")
        t1=perf_counter()
        theoretical_energy=energy.theoretical_energy(N,k,h)
        ttott+=perf_counter()-t1
        if(j%20==0):print(f"theoretical {ttott}")
        cose[j,i]=-(computed-theoretical_energy)/theoretical_energy*100
        #print(f"k={k} h={h} computed with exp: {computed} \n theoretical energy: {theoretical_energy}")
        #print(f"error: {cose}\n")

np.set_printoptions(suppress=True, precision=4)
print(cose)

import matplotlib.pyplot as plt

ks = npzfile["ks"]
hs = npzfile["hs"]

data = cose
if data.shape != (len(hs), len(ks)):
    data = data.T

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, origin="lower", aspect="auto",
               extent=[ks.min(), ks.max(), hs.min(), hs.max()],
               cmap="viridis")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Relative error (%)")
ax.set_xlabel("k")
ax.set_ylabel("h")
ax.set_title("Heatmap of relative errors (cose)")
plt.tight_layout()
plt.savefig("blabla.png")