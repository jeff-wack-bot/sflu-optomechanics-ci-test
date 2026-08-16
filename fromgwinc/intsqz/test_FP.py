"""
"""

import numpy as np
from wield.control.SFLU import SFLU, optics, nx2tikz
from gwinc.struct import Struct
from gwinc.noise.quantum_lib import adjoint, Vnorm_sq
from os import path

F_Hz = np.logspace(-1, 4, 3000)

def test_load_IFO():
    tpath = path.split(__file__)[0]
    ifo = Struct.from_file(path.join(tpath, '../Aplus/ifo.yaml'))
    print(ifo)
