# make_topology.py
import pickle
from qutip import sigmax, sigmaz, basis

H0 = 0.5 * sigmax()
H1 = 0.3 * sigmaz()
H = [H0, [H1, lambda t, args: np.sin(t)]]
psi0 = basis(2, 0)
c_ops = []

topology = {
    "H": H,
    "psi0": psi0,
    "c_ops": c_ops
}

with open("topology.pkl", "wb") as f:
    pickle.dump(topology, f)
