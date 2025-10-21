# qsim_runner

Checkpointable, backend-selectable QuTiP workflow manager for HPC and preemptible VMs.
Supports `mesolve` and propagator workflows. Optional `qutip_jax` backend for GPU.

## Features
- Single-file checkpoints per simulation name: `checkpoints/<sim_name>.cpt`
- Periodic trajectory dumps to `trajectories/` using `qutip.qsave` (`.qu`, with pickle fallback)
- Time-dependent and static Hamiltonians
- Backends: `qutip.mesolve` and `qutip.propagator`; optional `qutip_jax` if installed

## How it works (TL;DR)
- Provide a config JSON and a topology pickle.
- Topology contains an `initial_state` expression (evaluated with `basis`/`tensor`) and a Hamiltonian “factory” registered on `qutip` that returns either a static `Qobj` or a QuTiP time-dependent Hamiltonian list.
- Set a simulation name (used for the single checkpoint file).
- Run; resume picks up from `checkpoints/<sim_name>.cpt`.

## Quickstart: Two-level system (TLS) with checkpoint/resume

This minimal example shows a driven TLS:
H(t) = (Δ/2) σ_z + (Ω/2) cos(ω t) σ_x, starting from |0>.

1) Create topology and config
Save the following as `make_tls_topology.py` and run it once.

````python
import json, pickle
import numpy as np
import qutip

# Physics
Delta = 1.0       # detuning
Omega = 2.0       # Rabi amplitude
omega = 1.5       # drive frequency

# Operators
sx = qutip.sigmax()
sz = qutip.sigmaz()

# Time-dependent envelope for σx
def drive(t, args):
    return 0.5 * Omega * np.cos(omega * t)

# Factory returning a mesolve-compatible Hamiltonian list
def _get_H_factory():
    H0 = 0.5 * Delta * sz
    H1 = sx
    return [H0, [H1, drive]]

# Initial state expression (evaluated inside qsim with basis/tensor)
init_expr = "basis(2, 0)"
psi0 = qutip.basis(2, 0)

# Topology payload
topo = {
    "H": 0.5 * Delta * sz,        # placeholder; replaced by factory at runtime
    "psi0": psi0,                 # fallback if expression eval is skipped
    "initial_state": init_expr,   # string expression
    "hamiltonian": [("get_H", ["get_H"])],  # tells qsim to call qutip.get_H()
    "c_ops": [],
}

# Config
config = {
    "dt": 0.02,
    "t_final": 5.0,
    "checkpoint_interval": 50,
    "trajectory_interval": 100,
    "sim_name": "tls_demo",
    "use_jax": False
}

with open("qsim_config.json", "w") as f:
    json.dump(config, f, indent=2)
with open("tls_topology.pkl", "wb") as f:
    pickle.dump(topo, f)

print("Wrote [qsim_config.json](http://_vscodecontentref_/0) and tls_topology.pkl")
`````

Edit config.json (see README examples).

Run and resume
Use this runner, which registers the Hamiltonian factory on qutip before constructing the simulator.
```python

# run_tls.py
import os
import numpy as np
import qutip
from qsim_runner import QSimMesolve

# Must match the physics in make_tls_topology.py
Delta = 1.0
Omega = 2.0
omega = 1.5

sx = qutip.sigmax()
sz = qutip.sigmaz()

def drive(t, args):
    return 0.5 * Omega * np.cos(omega * t)

def _get_H_factory():
    H0 = 0.5 * Delta * sz
    H1 = sx
    return [H0, [H1, drive]]

# Register the factory before creating the simulator
setattr(qutip, "get_H", _get_H_factory)

sim = QSimMesolve("qsim_config.json", "tls_topology.pkl")
sim.set_simulation_name(sim.config.get("sim_name", "tls_demo"))

# First run (fresh)
if not os.path.exists(os.path.join("checkpoints", f"{sim._get_sim_name()}.cpt")):
    print("Starting fresh run...")
    sim.run(resume=False)
else:
    # Resume from checkpoint
    step, _ = sim._load_checkpoint()
    print(f"Resuming from step {step}...")
    sim.run(resume=True)

print("Done.")
```


## License
GPL-3.0-or-later

## Notabene

Heavily vibe-coded. Probably full of hallucinations. Need to debug. Session record:
https://chatgpt.com/share/68ed3565-ff44-800d-82a8-534841c184bc
