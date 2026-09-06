from __future__ import annotations

from time import perf_counter

import jax.numpy as jnp
import numpy as np

import mrmustard.math as mm
import mrmustard.lab as lab

mm.change_backend("jax")

from mrmustard import settings              # noqa: E402
from mrmustard.parameters import Variable   # noqa: E402
from mrmustard.training import Optimizer    # noqa: E402

settings.PROGRESSBAR = False

from qoptcraft.state import PureState, Fock  # noqa: E402
from qoptcraft.optimization import OptimizerResult  # noqa: E402


def _fock_state(fock: tuple[int, ...], cutoffs: list[int], modes: range | list[int] | None = None):
    """Multi-mode lab.Number product state for one Fock occupation pattern.

    `modes` defaults to `range(len(fock))`; pass an explicit sequence to place
    the pattern on a different (e.g. non-zero-based) set of mode indices.
    """
    modes = range(len(fock)) if modes is None else modes
    modes = list(modes)
    state = lab.Number(mode=modes[0], n=fock[0], cutoff=cutoffs[0] - 1)
    for mode, n, cutoff in zip(modes[1:], fock[1:], cutoffs[1:]):
        state = state.contract(lab.Number(mode=mode, n=n, cutoff=cutoff - 1))
    return state


class MrMustardOptimizer:
    """MrMustard Riemannian optimizer for heralded state prep."""

    def __init__(
        self,
        in_state: PureState,
        target_state: PureState,
        herald: Fock,
        *,
        cutoffs: list[int] | None = None,
        cost_type: str = "gubarev",
        alpha: float = 1e-3,
        beta: float = 4,
    ) -> None:
        self._parse_qoptcraft(in_state, target_state, herald, cutoffs)
        if cost_type not in ("gubarev", "piecewise"):
            raise ValueError(f"cost_type must be 'gubarev' or 'piecewise', got {cost_type!r}.")
        self.cost_type = cost_type
        self.alpha = alpha
        self.beta = beta

    def _cost_from_fid_prob(self, fid, prob):
        """fid is the *heralded* fidelity (already divided by prob), matching
        both cost() implementations below."""
        if self.cost_type == "piecewise":
            infid = 1 - fid
            return jnp.where(infid < self.min_infid, infid - self.lagrange_multiplier * prob, infid)
        return -(prob ** self.alpha) * (fid ** self.beta)

    def _parse_qoptcraft(
        self,
        in_state: PureState,
        target_state: PureState,
        herald: Fock,
        cutoffs: list[int] | None,
    ) -> None:
        """Parse qoptcraft PureStates into the MrMustard objects the cost needs."""
        self.modes = in_state.modes
        self.herald_pattern = herald.fock_states[0]

        if cutoffs is None:
            signal_modes = self.modes - herald.modes
            signal_cutoff = in_state.photons - herald.photons + 1
            base = [signal_cutoff] * signal_modes + [n + 1 for n in self.herald_pattern]
            # Widen per mode if in_state itself occupies that mode more than the
            # herald-conditioned bound above (the bound only constrains the output
            # after the interferometer, not the input pattern).
            max_input = [max(fock[i] for fock in in_state.fock_states) for i in range(self.modes)]
            cutoffs = [max(b, n + 1) for b, n in zip(base, max_input)]
        if len(cutoffs) != self.modes:
            raise ValueError(f"cutoffs must have length {self.modes} (in_state.modes), got {len(cutoffs)}.")
        self.cutoffs = cutoffs

        self.input_terms = [
            (_fock_state(fock, self.cutoffs), amp)
            for fock, amp in zip(in_state.fock_states, in_state.amplitudes)
        ]
        self.target_terms = list(zip(target_state.fock_states, target_state.amplitudes))

    def cost(self, unitary):
        """Gubarev cost function."""
        gate = lab.Interferometer(modes=tuple(range(self.modes)), unitary=unitary)
        for wire, cutoff in zip(gate.wires.output.ket, self.cutoffs):
            wire.fock_shape = cutoff
        state_vector = None
        for state, amp in self.input_terms:
            out = (state >> gate).fock_array(shape=self.cutoffs)
            state_vector = out * amp if state_vector is None else state_vector + out * amp
        signal = state_vector[(...,) + self.herald_pattern]   # fix the trailing herald modes
        prob = mm.sum(mm.abs(signal) ** 2)
        overlap = sum(amp * signal[fock] for fock, amp in self.target_terms)
        fid = mm.abs(overlap) ** 2 / (prob + 1e-16)
        return -(prob ** self.alpha) * (fid ** self.beta)

    def minimize(
        self,
        max_iter: int = 10000,
        unitary_lr: float = 0.1,
    ) -> OptimizerResult:

        """Riemannian GD from a random Haar-uniform starting unitary."""
        var = Variable(value=mm.random_unitary(self.modes), name="unitary", update_fn="update_unitary")
        opt = Optimizer(unitary_lr=unitary_lr)

        t0 = perf_counter()
        [var] = opt.minimize(self.cost, by_optimizing=[var], max_steps=max_iter)
        cost_array = [float(c) for c in opt.opt_history[1:]]

        return OptimizerResult(
            point=np.array(mm.asnumpy(var.value)),
            iterations=len(cost_array),
            time=perf_counter() - t0,
            cost_array=cost_array,
        )
