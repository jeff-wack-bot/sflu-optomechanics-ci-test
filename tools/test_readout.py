"""Unit checks for sflu.readout that the paper reproductions do not reach."""
import numpy as np

from sflu import readout
from sflu.lib import MatrixLib


def test_referred_psd_mixes_frequency_independent_inputs():
    # solve_ac returns (dim, dim) with no frequency axis for a path with no
    # frequency dependence; summing it with (N, dim, dim) paths must work.
    mlib = MatrixLib(nhom=0)
    N = 5
    T = {
        "sig": np.ones((N, 2, 1)),
        "vac": np.broadcast_to(mlib.Id, (N, 2, 2)),
        "loss": 0.1 * mlib.Id,
    }
    total, budget = readout.referred_psd(T, readout.quadrature(mlib, np.pi / 2), "sig")
    assert total.shape == (N,)
    assert np.allclose(budget["loss"], 0.01 * budget["vac"])


def test_squeezed_is_vacuum_at_zero_dB_and_squeezes_the_named_quadrature():
    mlib = MatrixLib(nhom=1)
    assert np.allclose(readout.squeezed(mlib, 0, 0.3), mlib.Id)
    V = readout.squeezed(mlib, 10, np.pi / 2)
    assert np.isclose(V[1, 1], 0.1) and np.isclose(V[0, 0], 10)
    assert np.allclose(V[2:, 2:], np.eye(2))       # the HOM stays in vacuum
