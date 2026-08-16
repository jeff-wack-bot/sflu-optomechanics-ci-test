"""
Interferometer parameters: the yaml sets, and the derived quantities.

Two distinct things live under this package, which used to sit in one
directory looking alike:

``sflu/params/ifo/*.yaml``
    IFO **parameter sets** -- `Ahat*`, `Aplus*`, `Asharp*`. Consumed by
    ``gwinc.load_budget``. They form an inheritance chain via ``+inherit``,
    resolved relative to each file, and terminating at gwinc's own built-in
    ``Aplus`` budget.

``sflu/params/standardize.py``
    The ifo Struct to derived-parameters step (``standardize_params``).

Serialized SFLU *graphs* also used to live beside these. They are a different
kind of file entirely and now sit with the model that loads them, in
``sflu/models/topologies/``.

Reach the parameter sets by name rather than by path, so callers do not have
to know where the directory is:

    >>> from sflu.params import load_ifo
    >>> ifo = load_ifo('AhatTest')
"""
import os

from .standardize import arm_gouyRT, standardize_params

IFO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ifo')


def ifo_path(name):
    """Absolute path of an IFO parameter set, by name.

    ``name`` is the stem, with or without the ``.yaml`` suffix. Use this when
    something needs the path itself -- notably ``gwinc.load_budget(..., freq=)``
    when a full reference *budget* is wanted rather than just the parameters.
    """
    if not name.endswith('.yaml'):
        name = name + '.yaml'
    path = os.path.join(IFO_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no IFO parameter set {name!r} in {IFO_DIR}; "
            f"available: {', '.join(sorted(available()))}"
        )
    return path


def available():
    """Names of the available IFO parameter sets."""
    return [f[:-len('.yaml')] for f in os.listdir(IFO_DIR) if f.endswith('.yaml')]


def load_ifo(name):
    """Load an IFO parameter set by name and return its ``ifo`` Struct.

    This resolves the ``+inherit`` chain. Note that it returns the parameters
    only: the ``Budget`` object ``gwinc.load_budget`` builds along the way is
    discarded, because no model uses it. See ``docs/GWINC_DEPENDENCY.md``.
    """
    import gwinc

    return gwinc.load_budget(ifo_path(name)).ifo


__all__ = [
    "IFO_DIR",
    "arm_gouyRT",
    "available",
    "ifo_path",
    "load_ifo",
    "standardize_params",
]
