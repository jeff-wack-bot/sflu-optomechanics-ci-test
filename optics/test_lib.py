import numpy as np
import sflu_components.lib as lib
import pytest


@pytest.mark.skip
def test_Mrotation():
    pass


def test_MrotationMM(pprint):
    mlib = lib.MatrixLib(nhom=3)
    Larr = np.arange(1, 4) * 0.01
    psi_rad = np.arange(1, 4) * 15 * np.pi/180
    M1 = mlib.MrotationMM(Larr, psi_rad)
    M1_inv = mlib.MrotationMM(Larr, psi_rad, inv=True)
    Id = np.eye(2)
    zzz = 0 * Id
    M00 = np.sqrt(1 - np.sum(Larr)) * Id
    M11 = np.sqrt(1 - Larr[0]) * Id
    M22 = np.sqrt(1 - Larr[1]) * Id
    M33 = np.sqrt(1 - Larr[2]) * Id
    M01 = -np.sqrt(Larr[0]) * lib.Mrotation2(psi_rad[0])
    M02 = -np.sqrt(Larr[1]) * lib.Mrotation2(psi_rad[1])
    M03 = -np.sqrt(Larr[2]) * lib.Mrotation2(psi_rad[2])
    M10 = np.sqrt(Larr[0]) * lib.Mrotation2(-psi_rad[0])
    M20 = np.sqrt(Larr[1]) * lib.Mrotation2(-psi_rad[1])
    M30 = np.sqrt(Larr[2]) * lib.Mrotation2(-psi_rad[2])
    M2 = np.block([
        [M00, M01, M02, M03],
        [M10, M11, zzz, zzz],
        [M20, zzz, M22, zzz],
        [M30, zzz, zzz, M33],
    ])
    assert np.all(M1 == M2)
    assert np.allclose(M1_inv @ M1, mlib.Id)


def test_promote():
    mlib = lib.MatrixLib(nhom=2)
    M1a = mlib.promote(2)
    M1b = 2 * mlib.Id

    mat2 = np.arange(1, 10).reshape(3, 3)
    M2a = mlib.promote(mat2)
    Id2 = np.eye(2)
    zz2 = 0 * Id2
    M2b = np.block([
        [1 * Id2, 2 * Id2, 3 * Id2],
        [4 * Id2, 5 * Id2, 6 * Id2],
        [7 * Id2, 8 * Id2, 9 * Id2],
    ])
    M3 = mlib.promote(M2a)

    arr = np.arange(1, 4)
    M3a = mlib.promote(arr)
    M3b = np.block([
        [1 * Id2, zz2, zz2],
        [zz2, 2 * Id2, zz2],
        [zz2, zz2, 3 * Id2],
    ])

    assert np.all(M1a == M1b)
    assert np.all(M2a == M2b)
    assert np.all(M3 == M2a)
    assert np.all(M3a == M3b)


def test_A():
    """
    Also tests mlib.block_diag
    """
    mlib = lib.MatrixLib(nhom=2)
    assert np.allclose(mlib.Ai @ mlib.A, mlib.Id)
