"""
Edges for SFLU graphs
"""
import numpy as np
from gwinc import const
from gwinc.struct import Struct
from .lib import MatrixLib, adjoint, block_diag

pi2i = 2j*np.pi

from wield.control import SISO, MIMO, ss_bare
import wield.control.ss_bare.design


class MirrorEdge:
    """
    Defines DC and AC edges for a simple mirror like a BasisMirror
    """
    def __init__(
            self,
            name,
            Thr=0,
            Lhr=0,
            Rar=0,
            lambda_m=1064e-9,
            mlib=MatrixLib(nhom=0),
            loss_in_transmission=False,
    ):
        self.name = name
        if not loss_in_transmission:
            self.t = mlib.promote(np.sqrt(Thr))
            self.l = mlib.promote(np.sqrt(Lhr))
            self.r = mlib.promote(np.sqrt(
                mlib.Id
                - mlib.promote(Thr)
                - mlib.promote(Lhr)
                - mlib.promote(Rar)
            ))
        else:
            self.t = mlib.promote(np.sqrt(Thr - Lhr))
            self.l = mlib.promote(np.sqrt(Lhr))
            self.r = mlib.promote(np.sqrt(1 - Thr))

        self.lambda_m = lambda_m
        self.mlib = mlib

    def _optic_edges(self):
        edge_map = {
            self.name + ".fr.r": -self.r,
            self.name + ".bk.r": +self.r,
            self.name + ".fr.t": self.t,
            self.name + ".bk.t": self.t,
            self.name + ".fr.l": self.l,
            self.name + ".bk.l": self.l,
        }
        return edge_map

    def edgesDC(self):
        """
        Returns the DC edge map dictionary
        """
        edge_map = self._optic_edges()
        # edge_map.update({
        #     self.name + '.fr.px': self.mlib.Id,
        #     self.name + '.bk.px': self.mlib.Id
        # })
        return edge_map

    def edgesAC(self, *args, **kwargs):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        resultsDC: the dictionary of DC results
        """
        edge_map = self._optic_edges()

        # # DC fields at the mirror faces
        # def get_fieldsDC(tp):
        #     try:
        #         return resultsDC[self.name + tp]
        #     except KeyError:
        #         return 0 * self.mlib.Id_v

        # fieldsDC_fr_i = get_fieldsDC(".fr.i.tp")
        # fieldsDC_bk_i = get_fieldsDC(".bk.i.tp")

        # # displacement to p (phase) quadrature
        # px = 4 * np.pi / self.lambda_m * self.r
        # px_fr = px @ self.mlib.Mrotation(np.pi/2) @ fieldsDC_fr_i
        # px_bk = px @ self.mlib.Mrotation(np.pi/2) @ fieldsDC_bk_i

        # edge_map.update({
        #     self.name + ".fr.px": px_fr,
        #     self.name + ".bk.px": px_bk,
        # })

        return edge_map

    def edgesACSS(self, *args, **kwargs):
        """
        Return the state space representation (or just D Matrix) of the edges for the AC calculation, rather than a frequency response array.
        """
        edge_map = self.edgesAC(*args, **kwargs)
        assert(__debug__)
        if __debug__:
            for k, e in edge_map.items():
                e = np.asarray(e)
                assert(len(e.shape) <= 2)
                assert(len(e) <= 40)
        return edge_map


class BSEdge:
    def __init__(
            self,
            name,
            Thr=0.5,
            Lhr=0,
            mlib=MatrixLib(nhom=0),
    ):
        self.name = name
        self.t = mlib.promote(np.sqrt(Thr))
        self.l = mlib.promote(np.sqrt(Lhr))
        self.r = mlib.promote(np.sqrt(
            mlib.Id
            - mlib.promote(Thr)
            - mlib.promote(Lhr)
        ))
        self.mlib = mlib

    def _optic_edges(self):
        edge_map = {
            self.name + ".fr.r": -self.r,
            self.name + ".bk.r": +self.r,
            self.name + ".t": self.t,
        }
        return edge_map

    def edgesDC(self):
        edge_map = self._optic_edges()
        return edge_map

    def edgesAC(self, *args, **kwargs):
        edge_map = self._optic_edges()
        return edge_map

    def edgesACSS(self, *args, **kwargs):
        """
        Return the state space representation (or just D Matrix) of the edges for the AC calculation, rather than a frequency response array.
        """
        edge_map = self.edgesAC(*args, **kwargs)
        assert(__debug__)
        if __debug__:
            for k, e in edge_map.items():
                e = np.asarray(e)
                assert(len(e.shape) <= 2)
                assert(len(e) <= 40)
        return edge_map



class LinkEdge:
    """
    Defines DC and AC edges for propagation links

    Parameters
    ----------
    name : str
      Name of the link
    L_m: float
      Macroscopic length of the link [m]
    detune_rad : float, optional
      Common microscopic detuning of all fields [rad], 0 by default
    gouy_rad : nhom element list of scalars or (N,) arrays, optional
      Gouy phases for each HOM [rad], all 0 by default
    MM_fr : scalar or (dim, dim) array, optional
      Mode matching basis transformation at the beginning of the link,
      (dim, dim) identity by default
    MM_to : scalar or (dim, dim) array, optional
      Mode matching basis transformation at the end of the link,
      (dim, dim) identity by default
    mlib : MatrixLib instance, optional
      MatrixLib to use for calculations, MatrixLib(nhom=0) by default
    """
    def __init__(
            self,
            name,
            L_m,
            detune_rad=0,
            gouy_rad=None,
            MM_fr=1,
            MM_to=1,
            mlib=MatrixLib(nhom=0),
    ):
        self.name = name
        self.L_m = L_m
        self.detune_rad = detune_rad
        self.MM_fr = mlib.promote(MM_fr)
        self.MM_to = mlib.promote(MM_to)
        self.mlib = mlib

        if gouy_rad is None:
            self.gouy_rad = np.zeros(mlib.nhom)
        else:
            if np.isscalar(gouy_rad):
                if mlib.nhom == 0:
                    self.gouy_rad = np.zeros(0)
                elif mlib.nhom == 1:
                    self.gouy_rad = np.array([gouy_rad])
                else:
                    raise ValueError('need gouy_phases for all HOMs')
            else:
                assert len(gouy_rad) == mlib.nhom
                self.gouy_rad = gouy_rad

    def _edges(self, Lmat):
        """
        This works equivalently in statespace
        """
        edge_map = {
            self.name: self.MM_to @ Lmat @ self.MM_fr,
        }
        return edge_map

    def edgesDC(self):
        """
        Returns the DC edge map dictionary
        """
        Lmat = self.mlib.Mrotation(self.detune_rad, *self.gouy_rad)
        return self._edges(Lmat)

    def edgesAC(self, F_Hz, *args, **kwargs):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        """
        delay = self.mlib.diag(np.exp(-pi2i * F_Hz * self.L_m / const.c))
        Lmat = delay @ self.mlib.Mrotation(self.detune_rad, *self.gouy_rad)
        return self._edges(Lmat)

    def edgesACSS(self, F_Hz, *args, **kwargs):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        """
        # TODO, need to compute the order more appropriately
        if self.L_m > 0:
            order = int(self.L_m / const.c * np.max(F_Hz) + 1)
            # print("Useorder: ", order)
            singledelay = SISO.design.delay_thiran_raw(
                delay_s=self.L_m / const.c,
                order=order + 2,
            ).asSS
            delayABCD = ss_bare.design.replicateSS(ss=singledelay.ss, dim=self.mlib.dim)

            Lmat = delayABCD @ self.mlib.Mrotation(self.detune_rad, *self.gouy_rad)
        else:
            Lmat = self.mlib.Mrotation(self.detune_rad, *self.gouy_rad)
        return self._edges(Lmat)


class SQZEdge:
    """
    Defines DC and AC edges for propagation links

    Parameters
    ----------
    name : str
      Name of the link
    L_m: float
      Macroscopic length of the link [m]
    detune_rad : float, optional
      Common microscopic detuning of all fields [rad], 0 by default
    gouy_rad : nhom element list of scalars or (N,) arrays, optional
      Gouy phases for each HOM [rad], all 0 by default
    MM_fr : scalar or (dim, dim) array, optional
      Mode matching basis transformation at the beginning of the link,
      (dim, dim) identity by default
    MM_to : scalar or (dim, dim) array, optional
      Mode matching basis transformation at the end of the link,
      (dim, dim) identity by default
    mlib : MatrixLib instance, optional
      MatrixLib to use for calculations, MatrixLib(nhom=0) by default
    """
    def __init__(
            self,
            name,
            sqzDB=0,
            sqzANGdeg=0,
            MM_fr=1,
            MM_to=1,
            mlib=MatrixLib(nhom=0),
            dual_unphysical=False,
    ):
        self.name = name
        self.sqzDB = sqzDB
        self.sqzANGrad = sqzANGdeg / 180 * np.pi
        self.MM_fr = mlib.promote(MM_fr)
        self.MM_to = mlib.promote(MM_to)
        self.dual_unphysical = dual_unphysical
        self.mlib = mlib

    def _edges(self):
        """
        This works equivalently in statespace
        """
        if not self.dual_unphysical:
            edge_map = {
                self.name:
                self.MM_to @
                self.mlib.Mrotation(self.sqzANGrad) @
                self.mlib.SQZc(10**(self.sqzDB/10), 10**(-self.sqzDB/10)) @
                self.mlib.Mrotation(-self.sqzANGrad) @ self.MM_fr,
            }
        else:
            edge_map = {
                self.name:
                self.MM_to @
                self.mlib.Mrotation(self.sqzANGrad) @
                self.mlib.SQZc(10**(self.sqzDB/10), 10**(self.sqzDB/10)) @
                self.mlib.Mrotation(-self.sqzANGrad) @ self.MM_fr,
            }
        return edge_map

    def edgesDC(self):
        """
        Returns the DC edge map dictionary
        """
        return self._edges()

    def edgesAC(self, F_Hz, *args, **kwargs):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        """
        return self._edges()

    def edgesACSS(self, F_Hz, *args, **kwargs):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        """
        return self._edges()


class RPMirrorEdge:
    """
    Defines DC and AC edges for mirrors with radiation pressure

    Parameters
    ----------
    name : str
      Name of the mirror
    Thr : float, optional
      Transmissivity of the HR surface, 0 by default
    Lhr : float, optional
      Loss of the HR surface, 0 by default
    Rar : float, optional
      Reflectivity of the AR surface, 0 by default
    suscept : callable, optional
      Mechanical susceptibility as a function of frequency in Hz, 0 by default
    lambda_m : float, optional
      Wavelength [m], 1064e-9 by default
    overlap : scalar, (nhom + 1, nhom + 1) matrix, or (dim, dim) matrix
      Matrix of overlap integrals between optical and mechanical modes,
      1 by default
    mlib : MatrixLib instance, optional
      MatrixLib to use for calculations, MatrixLib(nhom=0) by default

    Examples
    --------
    5 ppm transmissive free mass mirror of mass M_kg
    >>> suscept = lambda F_Hz: -1 / (M_kg * (2 * np.pi * F_Hz)**2)
    >>> mirr = RPMirrorEdge('M', Thr=5e-6, suscept=suscept)

    Bulk mode of a M_kg mass mirror with mechanical frequency Fm_Hz,
    mechanical Q of Qm, and mode overlap integrals Bnm for one HOM
    >>> def suscept(F_Hz):
            den = Fm_Hz**2 - F_Hz**2 + 1j * Fm_Hz * F_Hz / Qm
            return 1 / (M_kg * (2 * np.pi)**2 * den)
    >>> overlap = np.array([
            [B00, B01],
            [B01, B11],
        ])
    >>> mlib = MatrixLib(nhom=1)
    >>> mirr = RPMirrorEdge('M', suscept=suscept, overlap=overlap, mlib=mlib)
    """
    def __init__(
            self,
            name,
            Thr=0,
            Lhr=0,
            Rar=0,
            suscept=lambda x: np.zeros_like(x),
            # should be a wield.control.SISO.SISOStateSpace object
            suscept_ss=SISO.zpk([], [], 0).asSS,
            lambda_m=1064e-9,
            overlap=1,
            mlib=MatrixLib(nhom=0),
    ):
        self.name = name
        self.t = mlib.promote(np.sqrt(Thr))
        self.l = mlib.promote(np.sqrt(Lhr))
        self.r = mlib.promote(np.sqrt(
            mlib.Id
            - mlib.promote(Thr)
            - mlib.promote(Lhr)
            - mlib.promote(Rar)
        ))
        self.suscept = suscept
        self.suscept_ss = suscept_ss
        self.lambda_m = lambda_m
        self.overlap = mlib.promote(overlap)
        self.mlib = mlib

    def _optic_edges(self):
        edge_map = {
            self.name + ".fr.r": -self.r,
            self.name + ".bk.r": +self.r,
            self.name + ".fr.t": self.t,
            self.name + ".bk.t": self.t,
            self.name + ".fr.l": self.l,
            self.name + ".bk.l": self.l,
        }
        return edge_map

    def edgesDC(self):
        edge_map = self._optic_edges()
        # no radiation pressure at DC
        zz = {k: 0 * getattr(self.mlib, k) for k in ['Id_a', 'Id_v', 'Id_s']}
        edge_map.update({
            self.name + ".fr.xq.i": zz["Id_a"],
            self.name + ".fr.xq.o": zz["Id_a"],
            self.name + ".bk.xq.i": zz["Id_a"],
            self.name + ".bk.xq.o": zz["Id_a"],
            self.name + ".fr.px": zz["Id_v"],
            self.name + ".bk.px": zz["Id_v"],
        })
        return edge_map

    def edgesAC(self, F_Hz, resultsDC, *args, **kwargs):
        edge_map = self._optic_edges()

        # DC fields at the mirror faces
        def get_fieldsDC(tp):
            try:
                return resultsDC[self.name + tp]
            except KeyError:
                return 0 * self.mlib.Id_v

        fieldsDC_fr_i = get_fieldsDC(".fr.i.tp")
        fieldsDC_fr_o = get_fieldsDC(".fr.o.tp")
        fieldsDC_bk_i = get_fieldsDC(".bk.i.tp")
        fieldsDC_bk_o = get_fieldsDC(".bk.o.tp")

        # displacement to p (phase) quadrature
        px = 4 * np.pi / self.lambda_m * self.r @ self.overlap
        px_fr = px @ self.mlib.Mrotation(np.pi/2) @ fieldsDC_fr_i
        px_bk = px @ self.mlib.Mrotation(np.pi/2) @ fieldsDC_bk_i

        # mechanical susceptibility, reshaped for multiplication
        chi = self.suscept(F_Hz)
        if np.isscalar(chi):
            chi = chi * self.mlib.Id_s
        else:
            chi = chi.reshape((-1, 1, 1))

        # q (amplitude) quadrature to displacement
        def xq_port(fieldsDC):
            return 2 / const.c * chi * adjoint(self.overlap @ fieldsDC)

        xq_fr_i = +xq_port(fieldsDC_fr_i)
        xq_fr_o = +xq_port(fieldsDC_fr_o)
        xq_bk_i = -xq_port(fieldsDC_bk_i)
        xq_bk_o = -xq_port(fieldsDC_bk_o)

        edge_map.update({
            self.name + ".fr.xq.i": xq_fr_i,
            self.name + ".fr.xq.o": xq_fr_o,
            self.name + ".bk.xq.i": xq_bk_i,
            self.name + ".bk.xq.o": xq_bk_o,
            self.name + ".fr.px": px_fr,
            self.name + ".bk.px": px_bk,
        })
        return edge_map

    def edgesACSS(self, resultsDC, *args, **kwargs):
        edge_map = self._optic_edges()

        # DC fields at the mirror faces
        def get_fieldsDC(tp):
            try:
                return resultsDC[self.name + tp]
            except KeyError:
                return 0 * self.mlib.Id_v

        fieldsDC_fr_i = get_fieldsDC(".fr.i.tp")
        fieldsDC_fr_o = get_fieldsDC(".fr.o.tp")
        fieldsDC_bk_i = get_fieldsDC(".bk.i.tp")
        fieldsDC_bk_o = get_fieldsDC(".bk.o.tp")

        # displacement to p (phase) quadrature
        px = 4 * np.pi / self.lambda_m * self.r @ self.overlap
        px_fr = px @ self.mlib.Mrotation(np.pi/2) @ fieldsDC_fr_i
        px_bk = px @ self.mlib.Mrotation(np.pi/2) @ fieldsDC_bk_i

        # mechanical susceptibility, reshaped for multiplication
        chi = self.suscept_ss.ss

        # q (amplitude) quadrature to displacement
        def xq_port(fieldsDC):
            return chi @ (2 / const.c * adjoint(self.overlap @ fieldsDC))

        xq_fr_i = +xq_port(fieldsDC_fr_i)
        xq_fr_o = +xq_port(fieldsDC_fr_o)
        xq_bk_i = -xq_port(fieldsDC_bk_i)
        xq_bk_o = -xq_port(fieldsDC_bk_o)

        edge_map.update({
            self.name + ".fr.xq.i": xq_fr_i,
            self.name + ".fr.xq.o": xq_fr_o,
            self.name + ".bk.xq.i": xq_bk_i,
            self.name + ".bk.xq.o": xq_bk_o,
            self.name + ".fr.px": px_fr,
            self.name + ".bk.px": px_bk,
        })
        return edge_map
