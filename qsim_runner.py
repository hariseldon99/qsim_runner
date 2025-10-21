import os, json, pickle
import numpy as np
import qutip
from qutip import Qobj, basis, propagator, mesolve
try:
    import qutip_jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class QSimBase:
    def __init__(self, config_file, topo_file):
        self.config = json.load(open(config_file))
        with open(topo_file, "rb") as f:
            self.topo = pickle.load(f)
        self.H = self.topo["H"]
        self.state = self.topo["psi0"]
        self.c_ops = self.topo.get("c_ops", [])
        self.dt = self.config["dt"]
        self.t_final = self.config["t_final"]
        self.times = np.arange(0, self.t_final, self.dt)
        self.chk_int = self.config.get("checkpoint_interval", 100)
        self.trj_int = self.config.get("trajectory_interval", 50)
        self.check_dir = "checkpoints"
        self.traj_dir = "trajectories"
        os.makedirs(self.check_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.state = eval(self.topo["initial_state"], {"basis": basis})
        self.H = self._build_hamiltonian()
        self.use_jax = self.config.get("use_jax", False) and HAS_JAX
        self.method = self.config.get("method", "adams")

    def _build_hamiltonian(self):
        H_parts = []
        for expr, deps in self.topo["hamiltonian"]:
            locs = {d: getattr(qutip, d)() for d in deps}
            H_parts.append(eval(expr, locs))
        return sum(H_parts)
    
    def set_simulation_name(self, name):
        """Set a base simulation name (without extension) used for checkpoint files."""
        if name.endswith(".cpt"):
            name = name[:-4]
        object.__setattr__(self, "_sim_name", name)

    def _get_sim_name(self):
        # prefer explicit attribute, then config, then ask the user
        if hasattr(self, "_sim_name") and getattr(self, "_sim_name"):
            return getattr(self, "_sim_name")
        cfg_name = self.config.get("sim_name") if hasattr(self, "config") else None
        if cfg_name:
            if cfg_name.endswith(".cpt"):
                cfg_name = cfg_name[:-4]
            object.__setattr__(self, "_sim_name", cfg_name)
            return cfg_name
        # interactive fallback
        try:
            name = input("Enter simulation name for checkpoints (no extension): ").strip()
        except Exception:
            name = "simulation"
        if not name:
            name = "simulation"
        object.__setattr__(self, "_sim_name", name)
        return name

    def __getattribute__(self, name):
        # Intercept checkpoint methods to use a single named .cpt file per simulation
        if name in ("_save_checkpoint", "_load_checkpoint"):
            # avoid recursion by fetching needed attrs with object.__getattribute__
            def _save(step, state):
                sim = object.__getattribute__(self, "_get_sim_name")()
                check_dir = object.__getattribute__(self, "check_dir")
                os.makedirs(check_dir, exist_ok=True)
                path = os.path.join(check_dir, f"{sim}.cpt")
                with open(path, "wb") as f:
                    pickle.dump((step, state), f)

            def _load():
                sim = object.__getattribute__(self, "_get_sim_name")()
                check_dir = object.__getattribute__(self, "check_dir")
                path = os.path.join(check_dir, f"{sim}.cpt")
                if not os.path.exists(path):
                    # match previous behavior: return (0, self.state) when no checkpoint
                    return 0, object.__getattribute__(self, "state")
                with open(path, "rb") as f:
                    step, state = pickle.load(f)
                return step, state

            return _save if name == "_save_checkpoint" else _load
    
        return object.__getattribute__(self, name)
    
    def _save_checkpoint(self, step, state):
        path = os.path.join(self.check_dir, f"chk_{step:05d}.pkl")
        with open(path, "wb") as f:
            pickle.dump((step, state), f)

    def _load_checkpoint(self):
        files = sorted(os.listdir(self.check_dir))
        if not files:
            return 0, self.state
        last = files[-1]
        with open(os.path.join(self.check_dir, last), "rb") as f:
            step, state = pickle.load(f)
        return step, state

    def _save_trajectory(self, step, state):
        path = os.path.join(self.traj_dir, f"traj_{step:05d}.qobj")
        state.save(path)


class QSimMesolve(QSimBase):
    def run(self, resume=True):
        start, psi = self._load_checkpoint() if resume else (0, self.state)
        for i, t in enumerate(self.times[start:], start=start):
            if self.use_jax:
                solver = qutip_jax.mesolve
            else:
                solver = mesolve
            result = solver(self.H, psi, [t, t + self.dt], self.c_ops)
            psi = result.states[-1]
            if i % self.chk_int == 0:
                self._save_checkpoint(i, psi)
            if i % self.trj_int == 0:
                self._save_trajectory(i, psi)
        self.state = psi


class QSimPropagator(QSimBase):
    def run(self, resume=True):
        start, U_accum = self._load_checkpoint() if resume else (0, qutip.qeye(self.state.dims[0][0]))
        for i, t in enumerate(self.times[start:], start=start):
            if self.use_jax:
                solver = qutip_jax.propagator
            else:
                solver = propagator
            U = solver(self.H, [t, t + self.dt], self.c_ops)
            U_accum = U @ U_accum
            if i % self.chk_int == 0:
                self._save_checkpoint(i, U_accum)
            if i % self.trj_int == 0:
                state = U_accum @ self.state
                self._save_trajectory(i, state)
        self.final_propagator = U_accum


if __name__ == "__main__":
    # Example: N=8 sinusoidally driven transverse-field Ising model (TFIM)
    N = 8
    J = 1.0
    h0 = 0.5           # static offset of transverse field
    h_amp = 0.8        # drive amplitude
    omega = 2.0 * np.pi / 5.0

    # simulation config
    config = {
        "dt": 0.05,
        "t_final": 5.0,
        "checkpoint_interval": 20,
        "trajectory_interval": 50,
        "sim_name": "tfim_N8",
    }
    config_file = "qsim_config.json"
    topo_file = "tfim_topology.pkl"

    # build many-body operators
    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    qI = qutip.qeye(2)

    def pauli_sum(op):
        """Return sum_j (I x ... x op_j x ... x I)"""
        terms = []
        for j in range(N):
            ops = [qI] * N
            ops[j] = op
            terms.append(qutip.tensor(*ops))
        return sum(terms)

    # Ising interaction (nearest-neighbour, open chain)
    H_zz = sum(
        qutip.tensor(*([qI] * (i) + [sz, sz] + [qI] * (N - i - 2)))
        for i in range(N - 1)
    )
    H_zz = -J * H_zz

    # transverse field X sum
    H_x = -pauli_sum(sx)

    # time-dependent prefactor for H_x
    def drive(t, args):
        return h0 + h_amp * np.sin(omega * t)

    # make a zero-arg factory on the qutip module so _build_hamiltonian can call it
    def _get_H_factory():
        # return a list suitable for qutip.mesolve: [H_static, [H1, time_func]]
        return [H_zz, [H_x, drive]]

    setattr(qutip, "get_H", _get_H_factory)

    # initial state: all spins up in sz (|0> for each qubit). Build an expression
    # that uses only `basis` (it will be eval'd inside QSimBase with {"basis": basis}).
    init_expr = "basis(2,0)"
    for _ in range(N - 1):
        init_expr += ".tensor(basis(2,0))"

    # also build a concrete psi0 Qobj for inclusion in the pickle (used as fallback)
    psi0 = qutip.tensor(*([qutip.basis(2, 0)] * N))

    # topology dict expected by QSimBase
    topo = {
        "H": H_zz,  # a static placeholder (will be replaced by _build_hamiltonian)
        "psi0": psi0,
        "initial_state": init_expr,
        # provide a hamiltonian list where expr 'get_H' will be eval'd; deps tells
        # _build_hamiltonian to call qutip.get_H() and make it available in eval
        "hamiltonian": [("get_H", ["get_H"])],
        "c_ops": [],  # no collapse operators for this example
    }

    # save config and topology to files
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    with open(topo_file, "wb") as f:
        pickle.dump(topo, f)

    # run the mesolve simulation with checkpointing (start fresh)
    sim = QSimMesolve(config_file, topo_file)
    sim.set_simulation_name(config["sim_name"])
    sim.run(resume=False)