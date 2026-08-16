"""
Smoke test for the IFO parameter loader.

Was a print of a vendored copy of gwinc's Aplus parameters, via a relative
path with one ``..`` too many, so it had been failing for a long time. That
vendored copy turned out to carry kuns-fork values that nothing actually ran
against, and now lives in ``attic/ifo_packages/``. This exercises the real
loader instead.
"""
import numpy as np

from sflu.params import available, load_ifo


def test_load_IFO(pprint):
    ifo = load_ifo('AhatTest')

    # the +inherit chain must have resolved: these come from the base Aplus
    # parameter set, not from AhatTest.yaml itself
    assert ifo.Infrastructure.Length == 3995
    assert ifo.Laser.Wavelength > 0
    assert ifo.Optics.ITM.Transmittance > 0
    assert 'intSqueezer' in ifo

    pprint(f"parameter sets available: {sorted(available())}")
