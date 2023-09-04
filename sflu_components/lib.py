import numpy as np
from scipy.linalg import block_diag


def Minv(M):
    if np.ndim(M) < 2:
        return 1 / M
    else:
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
    """
    This only works for dim < 6, otherwise matrix_stack is unhappy
    """
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

    @property
    def nhom(self):
        """
        Number of HOMs
        """
        return self._nhom

    @property
    def dim(self):
        """
        Dimension of the vector space: 2 * (nhom + 1)
        """
        return 2 * (self.nhom + 1)

    @property
    def zeros(self):
        """
        (dim, dim) zero matrix
        """
        return np.zeros((self.dim, self.dim))

    @property
    def Id(self):
        """
        (dim, dim) identity matrix
        """
        return np.eye(self.dim)

    @property
    def Id_v(self):
        """
        (dim, 1) identity vector
        """
        return np.ones((self.dim, 1))

    @property
    def Id_a(self):
        """
        (1, dim) adjoint identity vector
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
        phi : float or (N,) array
          Common rotation angle [rad]
        *psi : list of floats or (N,) arrays, optional
          Extra Gouy phases for each HOM [rad]
          Defaults to zero if none are given. If any Gouy phases are specified
          they all must be specified

        Returns
        -------
        M : (dim, dim) or (N, dim, dim) matrix
          The rotation matrix

        Examples
        --------
        Rotate fundamental and two HOMs by pi/6
        >>> mlib = MatrixLib(nhom=2)
        >>> mlib.Mrotation(np.pi/6)

        Rotate by pi/6 and Gouy phases pi/4 and pi/2
        >>> mlib.Mrotation(np.pi/6, np.pi/4, np.pi/2)

        Rotate the fundamental by pi/6 and a single HOM by 30, 45, and 60 deg
        >>> mlib = MatrixLib(nhom=1)
        >>> psi = np.array([30, 45, 60]) * np.pi/180
        >>> mlib.Mrotation(np.pi/6, psi)
        """
        phis = [np.atleast_1d(phi)] * (self.nhom + 1)
        if len(psi) == 0:
            # psi = np.zeros(self.nhom)
            thetas = [np.zeros_like(phis[0])] * (self.nhom + 1)
        else:
            if len(psi) != self.nhom:
                raise ValueError(
                    'Number of Gouy phases not equal to the number of HOMs')
            thetas = [np.zeros_like(np.atleast_1d(psi[0]))] + list(psi)
        max_dim = max(len(phis[0]), len(thetas[0]))
        M = np.zeros((max_dim, self.dim, self.dim))
        for ii, (phi, theta) in enumerate(zip(phis, thetas)):
            inds = slice(2 * ii, 2 * ii + 2)
            M[..., inds, inds] = Mrotation2(phi + theta)
        return M.squeeze()

    def MrotationMM(self, L, psi, inv=False):
        """
        Mode mismatch matrix

        Currently only mismatch between the fundamental and HOMs is implemented.
        No conversion between HOMs is included.

        Parameters
        ----------
        L : float (for a single HOM) or (nhom,) array
          Power loss from the fundamental to each HOM
        psi : float (for a single HOM) or (nhom,) array
          Mismatch phase for each mismatch [rad]

        Returns
        -------
        M : (dim, dim) matrix
          The mismatch matrix
        """
        if self.nhom == 0:
            if L != 0:
                raise ValueError('need at least one HOM for mismatch')
            return self.Id

        L_arr = np.atleast_1d(L)
        psi_arr = np.atleast_1d(psi)
        for var in [L_arr, psi_arr]:
            if len(var) != self.nhom:
                raise ValueError('mismatch data needed for all HOMs')

        M = self.zeros
        M[:2, :2] = np.sqrt(1 - np.sum(L_arr)) * np.eye(2)
        for ii, (L, psi) in enumerate(zip(L_arr, psi_arr)):
            inds = slice(2 * (ii + 1), 2 * (ii + 2))
            M[:2, inds] = -np.sqrt(L) * Mrotation2(psi)
            M[inds, :2] = np.sqrt(L) * Mrotation2(-psi)
            M[inds, inds] = np.sqrt(1 - L) * np.eye(2)
        if inv is True:
            M = Minv(M)
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
        M : (dim, dim) or (N, dim, dim) array
        """
        # return matrix_stack_id(val, self.dim)
        return np.einsum('...,ij', val, self.Id)

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
        S : (dim, dim) array
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
        v : (dim, 1) array
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
        M : (N, dim, dim) array
        """
        M = self.Id
        M[:2, :2] = RPNK2(K)
        return M

    def block_diag(self, val, dtype=float):
        """
        Construct a block diagonal matrix
        """
        assert val.shape == (2, 2)
        M = np.zeros((self.dim, self.dim), dtype=dtype)
        for ii in range(self._nhom + 1):
            inds = slice(2 * ii, 2 * ii + 2)
            M[inds, inds] = val
        return M

    def promote(self, mat):
        """
        Promote lower dimensional matrices to the full HOM space

        Parameters
        ----------
        mat : scalar, (nhom + 1, nhom + 1) matrix, (dim, dim) matrix, or (nhom + 1,) array

        Returns
        -------
        M : (dim, dim) array
            * if mat is a scalar, matrix with mat along the diagonal
            * if mat is an (nhom + 1, nhom + 1) matrix, a (dim, dim) block matrix
              with each 2x2 the corresponding matrix element of mat on the diagonal
            * if mat is an (nhom + 1,) array, a (dim, dim) block diagonal matrix with
              each block the corresponding 2x2 diagonal matrix
            * mat if mat is a (dim, dim) matrix

        Examples
        --------
        >>> mlib = MatrixLib(nhom=1)
        >>> mat = np.array([
                      [1, 2],
                      [3, 4],
                  ])
        >>> mlib.promote(mat)
            np.array([
                [1, 0, 2, 0],
                [0, 1, 0, 2],
                [3, 0, 4, 0],
                [0, 3, 0, 4],
            ])
        >>> arr = np.array([1, 2])
        >>> mlib.promote(arr)
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 2],
            ])
        """
        if np.isscalar(mat):
            M = mat * self.Id
        elif mat.shape == (self.nhom + 1, self.nhom + 1):
            M = self.zeros.astype(mat.dtype)
            for (ri, ci), x in np.ndenumerate(mat):
                row = slice(2 * ri, 2 * ri + 2)
                col = slice(2 * ci, 2 * ci + 2)
                M[row, col] = x * np.eye(2)
        elif mat.shape == (self.dim, self.dim):
            M = mat
        elif mat.shape == (self.nhom + 1,):
            M = self.zeros.astype(mat.dtype)
            for ii, x in enumerate(mat):
                inds = slice(2 * ii, 2 * ii + 2)
                M[inds, inds] = x * np.eye(2)
        else:
            raise ValueError("incorrect dimensions")
        return M

    @classmethod
    def Minv(cls, M):
        """
        Matrix inverse
        """
        return Minv(M)

    @classmethod
    def transpose(cls, M):
        """
        Matrix transpose
        """
        return transpose(M)

    @classmethod
    def adjoint(cls, M):
        """
        Matrix adjoint
        """
        return adjoint(M)

    @classmethod
    def Vnorm_sq(cls, M):
        """
        The matrix norm squared: adjoint(M) @ M
        """
        return Vnorm_sq(M)

    @property
    def A(self):
        """
        A matrix converting between sidebands and 2-photon

        Returns
        -------
        A : (dim, dim) array
        """
        return self.block_diag(A2, dtype=complex)

    @property
    def Ai(self):
        """
        Inverse A matrix converting between sidebands and 2-photon

        Returns
        -------
        Ai : (dim, dim) array
        """
        return self.block_diag(A2i, dtype=complex)
