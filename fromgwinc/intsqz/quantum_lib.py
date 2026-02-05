'''Functions to calculate quantum noise

'''
from __future__ import division
import numpy as np
from ..struct import Struct


def Minv(M):
    return np.linalg.inv(M)


def transpose(M):
    return np.swapaxes(M, len(M.shape)-1, len(M.shape)-2)


def adjoint(M):
    return transpose(M).conjugate()


def Vnorm_sq(M):
    #perhaps there is a faster way to compute this?
    sq = M @ adjoint(M)
    assert(sq.shape[-2:] == (1, 1))
    return sq[..., 0, 0].real


def matrix_stack(arr, dtype = None, **kwargs):
    """
    This routing allows one to construct 2D matrices out of heterogeneously
    shaped inputs. it should be called with a list, of list of np.array objects
    The outer two lists will form the 2D matrix in the last two axis, and the
    internal arrays will be broadcasted to allow the array construction to
    succeed

    example

    matrix_stack([
        [np.linspace(1, 10, 10), 0],
        [2, np.linspace(1, 10, 10)]
    ])

    will create an array with shape (10, 2, 2), even though the 0, and 2
    elements usually must be the same shape as the inputs to an array.

    This allows using the matrix-multiply "@" operator for many more
    constructions, as it multiplies only in the last-two-axis. Similarly,
    np.linalg.inv() also inverts only in the last two axis.
    """
    Nrows = len(arr)
    Ncols = len(arr[0])
    vals = []
    dtypes = []
    for r_idx, row in enumerate(arr):
        assert(len(row) == Ncols)
        for c_idx, kdm in enumerate(row):
            kdm = np.asarray(kdm)
            vals.append(kdm)
            dtypes.append(kdm.dtype)

    #dt = np.find_common_type(dtypes, ())
    if dtype is None:
        dtype = np.result_type(*vals)

    #do a huge, deep broadcast of all values
    idx = 0
    bc = None
    while idx < len(vals):
        if idx == 0 or bc.shape == ():
            v = vals[idx:idx+32]
            bc = np.broadcast(*v)
            idx += 32
        else:
            v = vals[idx:idx+31]
            #including bc breaks broadcast unless shape is not trivial
            bc = np.broadcast(bc, *v)
            idx += 31

    if len(bc.shape) == 0:
        return np.array(arr)

    Marr = np.empty(bc.shape + (Nrows, Ncols), dtype = dtype, **kwargs)
    #print(Marr.shape)

    for r_idx, row in enumerate(arr):
        for c_idx, kdm in enumerate(row):
            Marr[..., r_idx, c_idx] = kdm
    return Marr


A2 = matrix_stack([
    [1,  1],
    [-1j, 1j],
]) / 2**.5


A2i = matrix_stack([
    [1, 1j],
    [1, -1j],
]) / 2**.5


id2 = np.eye(2)


A4 = matrix_stack([
    [1,  1,   0,  0],
    [-1j, 1j, 0,  0],
    [0,  0,   1,  1  ],
    [0,  0,   -1j, 1j],
]) / 2**.5


A4i = matrix_stack([
    [1, 1j, 0, 0],
    [1, -1j,  0, 0],
    [0, 0,   1, 1j],
    [0, 0,   1, -1j],
]) / 2**.5


id4 = np.eye(4)


A6 = matrix_stack([
    [1,  1,   0,  0,   0,  0],
    [-1j, 1j, 0,  0,   0,  0],
    [0,  0,   1,  1  , 0,  0],
    [0,  0,   -1j, 1j, 0,  0],
    [0,  0,   0,  0,   1,  1],
    [0,  0,   0,  0,   -1j, 1j],
]) / 2**.5


A6i = matrix_stack([
    [1, 1j, 0, 0,   0, 0],
    [1, -1j,  0, 0,   0, 0],
    [0, 0,   1, 1j, 0, 0],
    [0, 0,   1, -1j,  0, 0],
    [0, 0,   0, 0,   1, 1j],
    [0, 0,   0, 0,   1, -1j],
]) / 2**.5


id6 = np.eye(6)


def matrix_stack_id2(value, **kwargs):
    arr = [value] * 2
    arrs = []
    for idx, a in enumerate(arr):
        lst = [0] * len(arr)
        lst[idx] = a
        arrs.append(lst)
    return matrix_stack(arrs, **kwargs)


def LO_field2(angle):
    # field = np.exp(1j * angle)
    # return A2 @ matrix_stack([[field], [field.conjugate()]]) / 2**.5
    return matrix_stack([[np.sin(angle)], [np.cos(angle)]])


def matrix_stack_id4(value, **kwargs):
    arr = [value] * 4
    arrs = []
    for idx, a in enumerate(arr):
        lst = [0] * len(arr)
        lst[idx] = a
        arrs.append(lst)
    return matrix_stack(arrs, **kwargs)


def LO_field4(angle):
    # field = np.exp(1j * angle)
    # return A4 @ matrix_stack([[field], [field.conjugate()], [0], [0]]) / 2**.5
    return matrix_stack([[np.sin(angle)], [np.cos(angle)], [0], [0]])


def matrix_stack_id6(value, **kwargs):
    arr = [value] * 6
    arrs = []
    for idx, a in enumerate(arr):
        lst = [0] * len(arr)
        lst[idx] = a
        arrs.append(lst)
    return matrix_stack(arrs, **kwargs)


def LO_field6(angle):
    # field = np.exp(1j * angle)
    # return A6 @ matrix_stack([[field], [field.conjugate()], [0], [0], [0], [0]]) / 2**.5
    return matrix_stack([[np.sin(angle)], [np.cos(angle)], [0], [0], [0], [0]])

def SQZ2(sqzV, asqzV):
    return matrix_stack([
        [asqzV**0.5, 0,      ],
        [0,          sqzV**0.5],
    ])


def RPNK2(K):
    return matrix_stack([
        [1,  0],
        [-K, 1],
    ])


def SQZ4(sqzV, asqzV):
    return matrix_stack([
        [asqzV**0.5, 0,         0, 0],
        [0,          sqzV**0.5, 0, 0],
        [0,          0,         1, 0],
        [0,          0,         0, 1],
    ])


def RPNK4(K):
    return matrix_stack([
        [1,  0, 0, 0],
        [-K, 1, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def SQZ6(sqzV, asqzV):
    return matrix_stack([
        [asqzV**0.5, 0,         0, 0, 0, 0],
        [0,          sqzV**0.5, 0, 0, 0, 0],
        [0,          0,         1, 0, 0, 0],
        [0,          0,         0, 1, 0, 0],
        [0,          0,         0, 0, 1, 0],
        [0,          0,         0, 0, 0, 1],
    ])


def RPNK6(K):
    return matrix_stack([
        [1, 0, 0, 0, 0, 0],
        [-K, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])


###################
# mats_planewave
###################
def Mrotation2(theta, theta2 = 0):
    return matrix_stack([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


def Mrotation2MMqp(L, phi = 0, inv = False):
    assert(L == 0)
    return id2


def promote2(d, mat):
    if mat.shape[-2:] == (2, 2):
        return mat
    else:
        return mat[..., :2, :2]


mats_planewave = Struct(
    SQZ         = SQZ2,
    LO          = LO_field2,
    Mrotation   = Mrotation2,
    MrotationMM = Mrotation2MMqp,
    A           = A2,
    Ai          = A2i,
    Id          = id2,
    diag        = matrix_stack_id2,
    RPNK        = RPNK2,
    Minv        = Minv,
    promote     = promote2,
)


#####################
# mats_mode_mismatch
#####################
def Mrotation4(theta, theta2 = 0):
    return matrix_stack([
        [np.cos(theta), -np.sin(theta), 0,             0],
        [np.sin(theta), np.cos(theta),  0,             0],
        [0,             0,              np.cos(theta + theta2), -np.sin(theta + theta2)],
        [0,             0,              np.sin(theta + theta2), np.cos(theta + theta2)],
    ])


def Mrotation4MMqp(L, phi = 0, inv = False):
    c = (1-L)**0.5
    s = L**0.5
    if not inv:
        M = matrix_stack([
            [+c, +0, -s, -0],
            [+0, +c, -0, -s],
            [+s, +0, +c, +0],
            [+0, +s, +0, +c],
        ])
    else:
        M = matrix_stack([
            [+c, +0, +s, +0],
            [+0, +c, +0, +s],
            [-s, -0, +c, +0],
            [-0, -s, +0, +c],
        ])
    if phi == 0:
        return M
    else:
        return Mrotation4(0, phi) @ M @ Mrotation4(0, -phi)


def promote4(d, mat):
    if mat.shape[-2:] == (2, 2):
        new = np.zeros(mat.shape[:-2] + (4, 4), dtype = mat.dtype)
        new[..., 2, 2] = d
        new[..., 3, 3] = d
        new[..., :2, :2] = mat
        return new
    elif mat.shape[-2:] == (6, 6):
        raise NotImplementedError
    else:
        return mat


mats_mode_mismatch = Struct(
    SQZ         = SQZ4,
    LO          = LO_field4,
    Mrotation   = Mrotation4,
    MrotationMM = Mrotation4MMqp,
    A           = A4,
    Ai          = A4i,
    Id          = id4,
    diag        = matrix_stack_id4,
    RPNK        = RPNK4,
    Minv        = Minv,
    promote     = promote4,
)


##################
# mats_misalign
##################

def Mrotation4(theta, theta2 = 0):
    return matrix_stack([
        [np.cos(theta), -np.sin(theta), 0,             0],
        [np.sin(theta), np.cos(theta),  0,             0],
        [0,             0,              np.cos(theta + theta2), -np.sin(theta + theta2)],
        [0,             0,              np.sin(theta + theta2), np.cos(theta + theta2)],
    ])


def Mrotation4MMqp(L, phi = 0, inv = False):
    c = (1-L)**0.5
    s = L**0.5
    if not inv:
        M = matrix_stack([
            [+c, +0, -s, -0],
            [+0, +c, -0, -s],
            [+s, +0, +c, +0],
            [+0, +s, +0, +c],
        ])
    else:
        M = matrix_stack([
            [+c, +0, +s, +0],
            [+0, +c, +0, +s],
            [-s, -0, +c, +0],
            [-0, -s, +0, +c],
        ])
    if phi == 0:
        return M
    else:
        return Mrotation4(0, phi) @ M @ Mrotation4(0, -phi)


mats_misalign = Struct(
    SQZ = SQZ4,
    LO = LO_field4,
    Mrotation = Mrotation4,
    MrotationMM = Mrotation4MMqp,
    A = A4,
    Ai = A4i,
    Id = id4,
    diag = matrix_stack_id4,
    RPNK = RPNK4,
    Minv = Minv,
    #promote = promote4,
)


###################
# mats_mm_and_misalign
###################
def Mrotation6(theta, theta2 = 0):
    #TODO
    raise NotImplementedError()
    return matrix_stack([
        [np.cos(theta), -np.sin(theta), 0,             0],
        [np.sin(theta), np.cos(theta),  0,             0],
        [0,             0,              np.cos(theta + theta2), -np.sin(theta + theta2)],
        [0,             0,              np.sin(theta + theta2), np.cos(theta + theta2)],
    ])


def Mrotation6MMqp(L, phi = 0, inv = False):
    #TODO
    raise NotImplementedError()
    c = (1-L)**0.5
    s = L**0.5
    if not inv:
        M = matrix_stack([
            [+c, +0, -s, -0],
            [+0, +c, -0, -s],
            [+s, +0, +c, +0],
            [+0, +s, +0, +c],
        ])
    else:
        M = matrix_stack([
            [+c, +0, +s, +0],
            [+0, +c, +0, +s],
            [-s, -0, +c, +0],
            [-0, -s, +0, +c],
        ])
    if phi == 0:
        return M
    else:
        return Mrotation6(0, phi) @ M @ Mrotation6(0, -phi)

def promote6(d, mat):
    if mat.shape[-2:] == (2, 2):
        new = np.zeros(mat.shape[:-2] + (6, 6), dtype = mat.dtype)
        new[..., 2, 2] = d
        new[..., 3, 3] = d
        new[..., 4, 4] = d
        new[..., 5, 5] = d
        new[..., :2, :2] = mat
        return new
    elif mat.shape[-2:] == (4, 4):
        new = np.zeros(mat.shape[:-2] + (6, 6), dtype = mat.dtype)
        new[..., 4, 4] = d
        new[..., 5, 5] = d
        new[..., :4, :4] = mat
        return new
    else:
        return mat


mats_mm_and_misalign = Struct(
    SQZ         = SQZ6,
    LO          = LO_field6,
    Mrotation   = Mrotation6,
    MrotationMM = Mrotation6MMqp,
    A           = A6,
    Ai          = A6i,
    Id          = id6,
    diag        = matrix_stack_id6,
    RPNK        = RPNK6,
    Minv        = Minv,
    promote     = promote6
)
