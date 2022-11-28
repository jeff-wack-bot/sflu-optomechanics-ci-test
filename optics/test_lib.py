import numpy as np
import sflu_components.lib as lib
import pytest


@pytest.mark.skip
def test_Mrotation():
    pass


def test_MrotationMM(pprint):
    mlib = lib.MatrixLib(nhom=1)
    M1 = mlib.MrotationMM(0.01, np.pi/6, inv=False)
    M1inv = mlib.MrotationMM(0.01, np.pi/6, inv=True)
    assert np.allclose(M1inv @ M1, mlib.Id)

    mlib = lib.MatrixLib(nhom=2)
    # Larr = np.arange(1, 3) * 0.01
    Larr = np.array([0.01, 0.01])
    psi_rad = np.arange(1, 3) * 15 * np.pi/180
    pprint(Larr.shape, psi_rad.shape)
    M2 = mlib.MrotationMM(Larr, psi_rad, inv=np.zeros(2, dtype=bool))
    M2inv = mlib.MrotationMM(Larr, psi_rad, inv=np.ones(2, dtype=bool))
    pprint(M2)
    pprint(M2inv)
    pprint(M2inv @ M2)
    pprint(np.isclose(M2inv @ M2, mlib.Id))
    pprint(np.allclose(M2inv @ M2, mlib.Id))
    # assert np.allclose(M2inv @ M2, mlib.Id)
################################################################################
    # inv = np.array([True, False, False, True])
    # M3a = mlib.MrotationMM(Larr, psi_rad, inv)
    # Id = np.eye(2)
    # zzz = 0 * Id
    # M00 = np.sqrt(1 - np.sum(Larr)) * Id
    # M11 = np.sqrt(1 - Larr[0]) * Id
    # M22 = np.sqrt(1 - Larr[1]) * Id
    # M33 = np.sqrt(1 - Larr[2]) * Id
    # M44 = np.sqrt(1 - Larr[3]) * Id
    # M01 = np.sqrt(Larr[0]) * lib.Mrotation2(-psi_rad[0])
    # M02 = -np.sqrt(Larr[1]) * lib.Mrotation2(-psi_rad[1])
    # M03 = -np.sqrt(Larr[2]) * lib.Mrotation2(-psi_rad[2])
    # M04 = np.sqrt(Larr[3]) * lib.Mrotation2(-psi_rad[3])
    # M10 = -np.sqrt(Larr[0]) * lib.Mrotation2(psi_rad[0])
    # M20 = np.sqrt(Larr[1]) * lib.Mrotation2(psi_rad[1])
    # M30 = np.sqrt(Larr[2]) * lib.Mrotation2(psi_rad[2])
    # M40 = np.sqrt(Larr[3]) * lib.Mrotation2(psi_rad[3])
    # M3b = np.block([
    #     [M00, M01, M02, M03, M04],
    #     [M10, M11, zzz, zzz, zzz],
    #     [M20, zzz, M22, zzz, zzz],
    #     [M30, zzz, zzz, M33, zzz],
    #     [M40, zzz, zzz, zzz, M44],
    # ])
    # pprint(M3a.shape)
    # pprint(M3b.shape)
    # # assert np.allclose(M3a, M3b)

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
