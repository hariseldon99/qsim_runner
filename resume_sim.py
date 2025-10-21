import os
import numpy as np
import qutip
from qsim_runner import QSimMesolve

# Must match the parameters used in the original run
N = 8
J = 1.0
h0 = 0.5
h_amp = 0.8
omega = 2.0 * np.pi / 5.0

# Rebuild the Hamiltonian pieces and expose qutip.get_H so QSimBase can reconstruct H
sx = qutip.sigmax()
sz = qutip.sigmaz()
qI = qutip.qeye(2)

def pauli_sum(op):
    terms = []
    for j in range(N):
        ops = [qI] * N
        ops[j] = op
        terms.append(qutip.tensor(*ops))
    return sum(terms)

# Static ZZ interaction (open chain)
H_zz = sum(
    qutip.tensor(*([qI] * i + [sz, sz] + [qI] * (N - i - 2)))
    for i in range(N - 1)
)
H_zz = -J * H_zz

# Transverse X field sum
H_x = -pauli_sum(sx)

# Time-dependent prefactor for H_x
def drive(t, args):
    return h0 + h_amp * np.sin(omega * t)

# Zero-arg factory returning a mesolve-compatible Hamiltonian
def _get_H_factory():
    # [H_static, [H1, time_func]]
    return [H_zz, [H_x, drive]]

# Make it available to QSimBase._build_hamiltonian()
setattr(qutip, "get_H", _get_H_factory)

# Files produced by the original run
config_file = "qsim_config.json"
topo_file = "tfim_topology.pkl"

if __name__ == "__main__":
    # Resume from the single named checkpoint file (config.sim_name.cpt)
    sim = QSimMesolve(config_file, topo_file)
    # Optional: ensure the same simulation name is used
    sim.set_simulation_name(sim.config.get("sim_name", "tfim_N8"))
    # Resume=True (default) will load checkpoints/<sim_name>.cpt if present
    sim.run(resume=True)
    print("Resume complete.")