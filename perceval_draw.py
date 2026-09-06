"""Draw the beamsplitter / phase-shifter decomposition of a unitary with Perceval."""

from typing import Literal

from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

from qoptcraft.optical_elements.clements_decomp import clements_rectangle_angles


TWO_PI = 2 * np.pi


def perceval_decomposition(
    unitary: NDArray,
    shape: Literal["clements", "reck"] = "clements",
    draw: bool = True,
    save_path: str | Path | None = None,
    font_scale: float = 1.0,
    **opts,
):
    """Decompose a unitary into beam splitters and phase shifters and draw it with Perceval.

    References:
        [1] Clements et al., "An Optimal Design for Universal Multiport Interferometers",
            Optica 3, 1460 (2016). https://arxiv.org/abs/1603.08788
        [2] Reck et al., "Experimental realization of any discrete unitary operator",
            Phys. Rev. Lett. 73, 58 (1994).
    """
    try:
        import perceval as pcvl
        from perceval.components import BS, PS, Circuit
        from perceval.rendering.circuit import SymbSkin
    except ImportError as error:
        raise ImportError(
            "perceval is required for perceval_decomposition. "
            "Install it with `pip install perceval-quandela`."
        ) from error

    _register_tikz_colors()

    if shape == "clements":
        circuit = _clements_circuit(pcvl, unitary)
        default_skin = _clements_skin()  # labels in qoptcraft's convention
    elif shape == "reck":
        matrix = pcvl.Matrix(np.asarray(unitary, dtype=complex))
        circuit = Circuit.decomposition(
            matrix,
            BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi_tr")),
            phase_shifter_fn=PS,
            shape=pcvl.InterferometerShape.TRIANGLE,
        )
        default_skin = SymbSkin(compact_display=True)  # perceval's own decomposition
    else:
        raise ValueError(f"shape must be 'clements' or 'reck', got '{shape}'.")

    opts.setdefault("precision", 1e-2)
    opts.setdefault("nsimplify", False)
    opts.setdefault("skin", default_skin)

    with _scaled_text(font_scale):
        if save_path is not None:
            # 'none' keeps the labels as <text> in an SVG instead of tracing them into
            # glyph outlines, so they stay editable in Inkscape or Illustrator.
            with matplotlib.rc_context({"svg.fonttype": "none"}):
                pcvl.pdisplay_to_file(circuit, str(save_path), **opts)
        if draw:
            _show(pcvl.pdisplay(circuit, **opts))
    return circuit


@contextmanager
def _scaled_text(factor: float):
    """Enlarge every label perceval draws, by patching the canvases for the duration."""
    if factor == 1:
        yield
        return

    from perceval.rendering.canvas import Canvas

    def scale(add_text):
        def scaled_add_text(self, points, text, size, *args, **kwargs):
            return add_text(self, points, text, size * factor, *args, **kwargs)

        return scaled_add_text

    originals = {canvas: canvas.add_text for canvas in Canvas.__subclasses__()}
    for canvas, add_text in originals.items():
        canvas.add_text = scale(add_text)
    try:
        yield
    finally:
        for canvas, add_text in originals.items():
            canvas.add_text = add_text


def _clements_circuit(pcvl, unitary: NDArray):
    """Fill perceval's rectangular mesh with qoptcraft's Clements angles.

    The shift rides along as the beam splitter's own top-left phase instead of a separate
    ``PS`` in front of it. Same unitary, but one component per mesh slot rather than two,
    which is what keeps the drawing narrow.
    """
    from perceval.components import GenericInterferometer

    angles, output_phases = clements_rectangle_angles(unitary)

    def beam_splitter(idx: int):
        angle, shift = angles[idx]
        # Both wraps are exact: the block is built from cos(angle), sin(angle) and
        # exp(1j * shift), so each is 2pi-periodic in qoptcraft's convention.
        return pcvl.BS.Ry(theta=2 * (angle % TWO_PI), phi_tl=shift % TWO_PI)

    return GenericInterferometer(
        unitary.shape[0],
        beam_splitter,
        shape=pcvl.InterferometerShape.RECTANGLE,
        phase_shifter_fun_gen=lambda mode: pcvl.PS(output_phases[mode] % TWO_PI),
        phase_at_output=True,
    )


def _clements_skin(compact: bool = True):
    """A skin that labels the mesh in qoptcraft's Clements convention.

    Perceval reads a beam splitter half-angled -- reflectivity ``cos(theta / 2)`` against
    qoptcraft's ``cos(angle)`` -- so its ``theta`` is twice the angle the decomposition
    returns, and is halved back here. The top-left phase is qoptcraft's ``shift``, so it
    drops perceval's positional suffix and prints as plain ``phi``.
    """
    from perceval.rendering.circuit import SymbSkin
    from perceval.utils import format_parameters

    class ClementsSkin(SymbSkin):
        def _get_display_content(self, circuit) -> str:
            variables = dict(circuit.get_variables())
            if "theta" in variables:  # a beam splitter, not the output phase layer
                variables["theta"] = float(variables["theta"]) / 2
                variables["phi"] = variables.pop("phi_tl")
            return format_parameters(variables, self.precision, self.nsimplify)

    return ClementsSkin(compact_display=compact)


def _register_tikz_colors() -> None:
    """Teach perceval's LaTeX canvas a color its own skins paint with."""
    from perceval.rendering.canvas import latex_canvas

    latex_canvas.tikz_implemented_colors.setdefault(
        "lightsalmon", "\\definecolor{lightsalmon}{rgb}{1.0, 0.63, 0.48}"
    )


def _show(drawing) -> None:
    """Display whatever ``pcvl.pdisplay`` hands back instead of drawing itself."""
    if drawing is None:
        return
    try:
        from IPython.display import display
    except ImportError:
        print(drawing)
    else:
        display(drawing)
