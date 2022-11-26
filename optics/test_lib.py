import numpy as np
import sflu_components.lib as lib
import pytest


@pytest.mark.skip
def test_Mrotation():
    pass


def test_promote():
    mlib = lib.MatrixLib(nhom=2)
    M1a = mlib.promote(2)
    M1b = 2 * mlib.Id

    mat2 = np.arange(1, 10).reshape(3, 3)
    M2a = mlib.promote(mat2)
    Id2 = np.eye(2)
    M2b = np.block([
        [1 * Id2, 2 * Id2, 3 * Id2],
        [4 * Id2, 5 * Id2, 6 * Id2],
        [7 * Id2, 8 * Id2, 9 * Id2],
    ])
    M3 = mlib.promote(M2a)

    assert np.all(M1a == M1b)
    assert np.all(M2a == M2b)
    assert np.all(M3 == M2a)


def test_A():
    """
    Also tests mlib.block_diag
    """
    mlib = lib.MatrixLib(nhom=2)
    assert np.allclose(mlib.Ai @ mlib.A, mlib.Id)
