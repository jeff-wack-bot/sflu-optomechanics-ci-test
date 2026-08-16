from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE

from gwinc.noise.quantum2 import (
    Quantum,
    QuantumRelShotNoise,
    QuantumRelGamma,
)


class Classical(nb.Budget):
    noises = [
        Seismic,
        Newtonian,
        SuspensionThermal,
        CoatingBrownian,
        CoatingThermoOptic,
        SubstrateBrownian,
        SubstrateThermoElastic,
        ExcessGas,
    ]

    plot_style = PLOT_STYLE

class Aplus(nb.Budget):

    name = 'A+'

    noises = [
        Quantum,
    ]
    noises_forward = [
        Classical
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
