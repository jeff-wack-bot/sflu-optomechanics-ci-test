import numpy as np


def Minv(M):
    return np.linalg.inv(M)


def transpose(M):
    return np.swapaxes(M, len(M.shape) - 1, len(M.shape) - 2)


def adjoint(M):
    return transpose(M).conjugate()


def Vnorm_sq(M):
    #perhaps there is a faster way to compute this?
    sq = adjoint(M) @ M
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


def matrix_stack_id(value, dim, **kwargs):
    arr = [value] * dim
    arrs = []
    for idx, a in enumerate(arr):
        lst = [0] * len(arr)
        lst[idx] = a
        arrs.append(lst)
    return matrix_stack(arrs, **kwargs)


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


A2 = matrix_stack([
    [1,  1],
    [-1j, 1j],
]) / 2**.5


A2i = matrix_stack([
    [1, 1j],
    [1, -1j],
]) / 2**.5


def Mrotation2(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return matrix_stack([
        [c, -s],
        [s, c],
    ])


class MatrixLib:
    """
    Matrix library for optical fields with arbitrary HOMs

    Parameters
    ----------
    nhom : int, optional
      Number of HOMs (Default: 0)
    """
    def __init__(self, nhom=0):
        self._nhom = nhom
        self._dim = 2 * (1 + nhom)

    @property
    def zeros(self):
        """
        (ndim, ndim) zero matrix
        """
        return np.zeros((self._dim, self._dim))

    @property
    def Id(self):
        """
        (ndim, ndim) identity matrix
        """
        return np.eye(self._dim)

    @property
    def Id_v(self):
        """
        (ndim, 1) identity vector
        """
        return np.ones((self._dim, 1))

    @property
    def Id_a(self):
        """
        (1, ndim) adjoint identity vector
        """
        return adjoint(self.Id_v)

    @property
    def Id_s(self):
        """
        (1, 1) scalar identity
        """
        return np.eye(1)

    def Mrotation(self, phi, *psi):
        """
        Rotation matrix with optional Gouy phases for HOMs

        Parameters
        ----------
        phi : float
          Common rotation angle [rad]
        *psi : list of floats, optional
          Extra Gouy phases for each HOM [rad]
          Defaults to zero if none are given. If any Gouy phases are specified
          they all must be specified

        Returns
        -------
        M : (ndim, ndim) matrix
          The rotation matrix

        Examples
        --------
        Rotate fundamental and two HOMs by pi/6
        >>> mlib = MatrixLib(nhom=2)
        >>> mlib.Mrotation(np.pi/6)

        Rotate by pi/6 and Gouy phases pi/4 and pi/2
        >>> mlib.Mrotation(np.pi/6, np.pi/4, np.pi/2)
        """
        if len(psi) == 0:
            psi = np.zeros(self._nhom)
        else:
            assert len(psi) == self._nhom
        thetas = np.hstack(([0], psi)) + phi
        M = self.zeros
        for ii, theta in enumerate(thetas):
            M[(2 * ii):(2 * ii + 2), (2 * ii):(2 * ii + 2)] = Mrotation2(theta)
        return M

    def diag(self, val):
        """
        Matrix with the same value along the diagonals

        Parameters
        ----------
        val : scalar or (N,) array
          Value along the diagonal

        Returns
        -------
        M : (ndim, ndim) or (N, ndim, ndim) array
        """
        return matrix_stack_id(val, self._dim)

    def SQZ(self, sqzV, asqzV):
        """
        Squeeze matrix

        Parameters
        ----------
        sqzV : float
          Squeezing variance
        asqzV : float
          Antisqueezing variance

        Returns
        -------
        S : (ndim, ndim) array
          Squeeze matrix for the fundamental mode

        Examples
        --------
        6 dB squeezing and 15 dB anti-squeezing
        >>> mlib.SQZ(10**(-6/10), 10**(15/10))
        """
        M = self.Id
        M[:2, :2] = SQZ2(sqzV, asqzV)
        return M

    def LO(self, phi):
        """
        LO for the fundamental

        The the phase quadrature is 0 deg and amplitude is 90 deg

        Parameters
        ----------
        phi : float
          LO angle [rad]

        Returns
        -------
        v : (ndim, 1) array
        """
        M = 0 * self.Id_v
        M[:2, 0] = [np.sin(phi), np.cos(phi)]
        return M

    def RPNK(self, K):
        """
        Radiation pressure noise matrix

        Parameters
        ----------
        K : (N,) array
          Optomechanical coupling

        Returns
        -------
        M : (N, ndim, ndim) array
        """
        M = self.Id
        M[:2, :2] = RPNK2(K)
        return M

    @classmethod
    def Minv(cls, M):
        """
        Matrix inverse
        """
        return Minv(M)

    @property
    def A(self):
        """
        A matrix converting between sidebands and 2-photon

        Returns
        -------
        A : (ndim, ndim) array
        """
        M = np.zeros((self._dim, self._dim), dtype=complex)
        for ii in range(self._nhom + 1):
            M[(2 * ii):(2 * ii + 2), (2 * ii):(2 * ii + 2)] = A2
        return M

    @property
    def Ai(self):
        """
        Inverse A matrix converting between sidebands and 2-photon

        Returns
        -------
        Ai : (ndim, ndim) array
        """
        M = np.zeros((self._dim, self._dim), dtype=complex)
        for ii in range(self._nhom + 1):
            M[(2 * ii):(2 * ii + 2), (2 * ii):(2 * ii + 2)] = A2i
        return M
