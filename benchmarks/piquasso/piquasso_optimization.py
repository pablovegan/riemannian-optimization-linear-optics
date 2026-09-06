from __future__ import annotations
import warnings
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import piquasso as pq
from numpy.exceptions import ComplexWarning
from scipy.optimize import minimize

from qoptcraft.state import PureState
from qoptcraft.math import haar_random_unitary
from qoptcraft.optimization import OptimizerResult
from piquasso.decompositions.clements import (
    get_interferometer_from_weights,
    get_weights_from_interferometer,
)


jax.config.update("jax_enable_x64", True)

warnings.filterwarnings(
    "ignore",
    category=ComplexWarning,
    message="Casting complex values to real discards the imaginary part",
)


def haar_random_clements_params(modes: int, rng=None) -> np.ndarray:
    """Haar-random initial params for Piquasso's native Clements parametrization."""
    rng = np.random.default_rng(rng)
    unitary = haar_random_unitary(modes, seed=rng)
    weights = get_weights_from_interferometer(unitary, pq.NumpyConnector())
    return np.asarray(weights, dtype=np.float64)


class PiquassoOptimizer:
    """
    Piquasso Optimizer utilizing native PostSelection and single-step JAX Unitaries.
    """
    def _parse_qoptcraft(
        self,
        in_state: PureState,
        target_state: PureState,
        herald: PureState,
    ) -> None:
        """Parse qoptcraft PureStates into the Piquasso objects the loss needs."""
        assert len(in_state.fock_states) == 1, "in_state must be a single Fock basis state."
        assert len(herald.fock_states) == 1, "Native post-selection requires a single herald Fock pattern."

        self.modes = in_state.modes
        self.cutoff = in_state.photons + 1

        in_fock = in_state.fock_states[0]
        herald_fock = herald.fock_states[0]
        self.herald_modes = tuple(range(target_state.modes, self.modes))

        # Convert the target from PureState to Piquasso state vector
        sim_sys = pq.PureFockSimulator(target_state.modes, pq.Config(cutoff=self.cutoff))

        with pq.Program() as prog:
            for t_fock, t_amp in zip(target_state.fock_states, target_state.amplitudes):
                pq.Q(all) | pq.StateVector(list(t_fock)) * complex(t_amp)

        target_sys_state = sim_sys.execute(prog).state
        target_sys_state.normalize()
        self.sys_target_sv_jax = jnp.array(target_sys_state.state_vector)

        self.init_instr = [pq.StateVector(list(in_fock))]
        self.post_select_instr = pq.PostSelectPhotons(
            photon_counts=tuple(herald_fock)
        ).on_modes(*self.herald_modes)

    def __init__(
        self,
        in_state: PureState,
        target_state: PureState,
        herald: PureState,
        *,
        alpha: float = 1e-3,
        beta: float = 4,
    ) -> None:
        self._parse_qoptcraft(in_state, target_state, herald)

        # Setup the JAX Piquasso Simulator
        self.connector = pq.JaxConnector()
        self.sim_jax = pq.PureFockSimulator(
            self.modes,
            pq.Config(cutoff=self.cutoff),
            connector=self.connector,
        )

        # Define the JIT-compiled Loss Function
        @jax.jit
        def _loss(params):
            U = get_interferometer_from_weights(params, self.modes, self.connector, dtype=jnp.complex128)

            # Piquasso's execute_instructions is the fastest way to bypass Program() tracing in JAX
            post_state = self.sim_jax.execute_instructions(
                self.init_instr + [pq.Interferometer(U), self.post_select_instr]
            ).state.state_vector

            prob = jnp.sum(jnp.abs(post_state) ** 2)
            fid = jnp.abs(jnp.vdot(self.sys_target_sv_jax, post_state)) ** 2 / (prob + 1e-16)

            return -(prob ** alpha) * (fid ** beta)

        self.loss_fn = _loss
        self.loss_and_grad = jax.jit(jax.value_and_grad(_loss))

    def minimize(self, max_iter: int = 5000) -> OptimizerResult:
        """Scipy L-BFGS-B using JAX gradients."""
        params0 = haar_random_clements_params(self.modes)
        cost_array = []
        n_iters = [0]

        def objective(p):
            loss, grad = self.loss_and_grad(jnp.array(p, dtype=jnp.float64))
            cost_array.append(float(loss))
            return float(loss), np.array(grad, dtype=np.float64)

        t0 = perf_counter()
        opt = minimize(
            objective, params0, method="BFGS", jac=True,
            callback=lambda _: n_iters.__setitem__(0, n_iters[0] + 1),
            options={"maxiter": max_iter, "gtol": 1e-8},
        )

        U_final = get_interferometer_from_weights(
            jnp.array(opt.x, dtype=jnp.float64), self.modes, self.connector, dtype=jnp.complex128
        )
        return OptimizerResult(
            point=np.array(U_final),
            iterations=n_iters[0],
            time=perf_counter() - t0,
            cost_array=cost_array,
        )
