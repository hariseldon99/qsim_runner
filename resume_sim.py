import os
import argparse
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

# Make it available to QSimBase._build_hamiltonian() BEFORE constructing the simulator
setattr(qutip, "get_H", _get_H_factory)

def main():
    parser = argparse.ArgumentParser(description="Resume or start a qsim run.")
    parser.add_argument("--config", default="qsim_config.json", help="Path to config JSON")
    parser.add_argument("--topo", default="tfim_topology.pkl", help="Path to topology pickle")
    parser.add_argument("--sim-name", default=None, help="Override simulation name used for checkpoints")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (do not resume)")
    args = parser.parse_args()

    sim = QSimMesolve(args.config, args.topo)

    # Determine sim name: CLI > config.sim_name > default
    sim_name = args.sim_name or sim.config.get("sim_name") or "simulation"
    sim.set_simulation_name(sim_name)

    resume = not args.fresh
    cpt_path = os.path.join("checkpoints", f"{sim_name}.cpt")

    if resume and not os.path.exists(cpt_path):
        print(f"No checkpoint found at {cpt_path}; starting fresh.")
        resume = False
    elif resume:
        # Show the checkpoint step we will resume from
        step, _state = sim._load_checkpoint()
        print(f"Resuming '{sim_name}' from step {step} ({cpt_path}).")

    sim.run(resume=resume)
    print("Run complete.")

if __name__ == "__main__":
    main()