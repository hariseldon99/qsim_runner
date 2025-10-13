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
