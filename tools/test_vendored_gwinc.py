"""
The vendored half of gwinc must keep agreeing with gwinc itself.

``sflu.params.load_ifo`` used to be ``gwinc.load_budget(path).ifo``; since
Stage 6 it is a pure YAML ``+inherit`` merge onto the vendored
``sflu/_vendor/gwinc/ifo/Aplus.yaml``. This pins the two together for every
parameter set, so if someone edits the vendored base file, or a parameter set
starts relying on a gwinc feature the vendored loader lacks, it fails here
rather than as a quiet shift in some budget.

Skips when gwinc is not installed: it is an optional dependency now, needed
only for the reference curves some examples draw.
"""
import numpy as np
import pytest

from sflu.params import available, ifo_path, load_ifo
from sflu._vendor.gwinc import noises as vendored_noises

gwinc = pytest.importorskip("gwinc")


def _flat(struct):
    return dict(struct.walk())


def _same(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return x == y
    return np.array_equal(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                          equal_nan=True)


@pytest.mark.parametrize("name", sorted(available()))
def test_load_ifo_matches_gwinc(name):
    ref = _flat(gwinc.load_budget(ifo_path(name)).ifo)
    got = _flat(load_ifo(name))
    assert ref.keys() == got.keys()
    differ = [k for k in ref if not _same(ref[k], got[k])]
    assert not differ, f"{name}: {differ}"


@pytest.mark.parametrize("name", sorted(available()))
def test_ifo_power_matches_gwinc(name):
    from gwinc.ifo.noises import ifo_power

    ifo = load_ifo(name)
    assert _flat(vendored_noises.ifo_power(ifo)) == _flat(ifo_power(ifo))


def test_dhdl_matches_gwinc():
    from gwinc.ifo.noises import dhdl

    f = np.geomspace(1, 1e5, 200)
    for ref, got in zip(dhdl(f, 4e3), vendored_noises.dhdl(f, 4e3)):
        assert np.array_equal(ref, got)
