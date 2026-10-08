"""
The small, deterministic half of pygwinc that the models depend on.

Vendored from pygwinc 0.6.2 (public domain, see ``LICENSE``) in Stage 6 of
``REFACTOR_PLAN.md``. The reasoning is in ``docs/GWINC_DEPENDENCY.md``; in
short, every model used gwinc for four things only:

``Struct``            attribute-access dict, and the YAML loader
``const``             ``c`` and ``hbar``
``noises``            ``ifo_power``, ``dhdl``, ``arm_cavity``
``load_struct``       what ``gwinc.load_budget(path).ifo`` actually did: a
                      YAML ``+inherit`` merge (the ``Budget`` it also built
                      was thrown away)

plus one data file, ``ifo/Aplus.yaml``, the base of every parameter set's
inherit chain. Before this, ``+inherit: Aplus`` meant "whatever pygwinc happens
to be installed", and the two forks in circulation disagree on the arm mirror
curvatures. Now it means this file, and changing it shows up in ``git log``.

What is *not* here: gwinc's noise budget engine (~6,000 lines across nine
noise disciplines). It is only used to draw reference curves under some
plots, and stays an optional dependency for that.

``struct.py``, ``const.py`` and ``noises.py`` are kept byte-comparable with
upstream apart from imports; each says exactly what was changed at its top.
"""
import os

from . import const
from .struct import Struct

BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ifo')


def builtin_path(name):
    """Path of a vendored base parameter set, e.g. ``'Aplus'``."""
    path = os.path.join(BUILTIN_DIR, name + '.yaml')
    if not os.path.exists(path):
        available = sorted(f[:-5] for f in os.listdir(BUILTIN_DIR))
        raise FileNotFoundError(
            f"{name!r} is not a vendored base parameter set "
            f"(have: {', '.join(available)}). gwinc resolved such names to its "
            f"own built-in budgets; vendor the one you need into {BUILTIN_DIR}."
        )
    return path


def load_struct(path):
    """Load a parameter file, resolving its ``+inherit`` chain.

    Reproduces ``gwinc.load_budget(path).ifo`` exactly, without building the
    budget: a relative ``+inherit`` resolves against the file that names it,
    a bare name such as ``Aplus`` resolves to the vendored base set, and the
    child is merged over its parent with gwinc's own rules (``<unset>``
    clears a key; scalars are not merged into).
    """
    path = os.fspath(path)
    ifo = Struct.from_file(path, _pass_inherit=True)
    inherit = ifo.get('+inherit', None)
    if inherit is None:
        return ifo
    del ifo['+inherit']

    rel_path = os.path.join(os.path.dirname(path), inherit)
    if os.path.splitext(inherit)[1] in Struct.STRUCT_EXT or os.path.exists(rel_path):
        parent = load_struct(rel_path)
    else:
        parent = Struct.from_file(builtin_path(inherit))

    parent.update(
        ifo,
        overwrite_atoms=False,
        clear_test=lambda v: isinstance(v, str) and v == '<unset>',
    )
    return parent


__all__ = ["Struct", "builtin_path", "const", "load_struct"]
