import numpy as np
import scipy.constants as scc
from gwinc.struct import Struct
from .lib import MatrixLib, adjoint

pi2i = 2j*np.pi


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
    ):
        Rhr = 1 - (Thr + Lhr)
        Tar = 1 - Rar

        self.name = name
        self.t = np.sqrt(Thr)
        self.r = np.sqrt(Rhr)

        self.lambda_m = lambda_m
        self.mlib = mlib

    def _optic_edges(self):
        t = self.mlib.diag(self.t)
        r = self.mlib.diag(self.r)
        edge_map = {
            self.name + '.fr.t': t,
            self.name + '.bk.t': t,
            self.name + '.fr.r': -r,
            self.name + '.bk.r': +r,
        }
        return edge_map

    def edgesDC(self):
        """
        Returns the DC edge map dictionary
        """
        edge_map = self._optic_edges()
        edge_map.update({self.name + '.fr.px': self.mlib.Id})
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        resultsDC: the dictionary of DC results
        """
        edge_map = self._optic_edges()
        try:
            fieldsDC = resultsDC[self.name + '.fr.i.tp']
            rtW_m = 4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC
            edge_map.update({self.name + '.fr.px': rtW_m})
        except KeyError:
            pass
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
        self.r = np.sqrt(1 - Thr - Lhr)
        self.t = np.sqrt(Thr)
        self.mlib = mlib

    def _optic_edges(self):
        t = self.mlib.diag(self.t)
        r = self.mlib.diag(self.r)
        edge_map = {
            self.name + ".fr.r": -r,
            self.name + ".bk.r": +r,
            self.name + ".t": t,
        }
        return edge_map

    def edgesDC(self):
        edge_map = self._optic_edges()
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._optic_edges()
        return edge_map


class LinkEdge:
    """
    Defines DC and AC edges for propagation links
    """
    def __init__(
            self,
            name,
            L_m,
            detune_rad=0,
            mlib=MatrixLib(nhom=0),
    ):
        self.name = name
        self.L_m = L_m
        self.detune_rad = detune_rad
        self.mlib = mlib

    def _edges(self, Lmat):
        edge_map = {
            self.name: Lmat,
        }
        return edge_map

    def edgesDC(self):
        """
        Returns the DC edge map dictionary
        """
        Lmat = self.mlib.Mrotation(self.detune_rad)
        return self._edges(Lmat)

    def edgesAC(self, F_Hz, resultsDC):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        """
        delay = self.mlib.diag(np.exp(-pi2i * F_Hz * self.L_m / scc.c))
        Lmat = delay @ self.mlib.Mrotation(self.detune_rad)
        return self._edges(Lmat)


class RPMirrorEdge:
    def __init__(
            self,
            name,
            Thr=0,
            Lhr=0,
            Rar=0,
            suscept_m_N=lambda x: np.zeroslike(x),
            lambda_m=1064e-9,
            overlap=1,
            mlib=MatrixLib(nhom=0),
    ):
        self.name = name
        self.t = np.sqrt(Thr)
        self.r = np.sqrt(1 - Thr - Lhr - Rar)
        self.suscept_m_N = suscept_m_N
        self.lambda_m = lambda_m
        self.overlap = overlap
        self.mlib = mlib

    def _optic_edges(self):
        t = self.mlib.diag(self.t)
        r = self.mlib.diag(self.r)
        edge_map = {
            self.name + ".fr.r": -r,
            self.name + ".bk.r": +r,
            self.name + ".fr.t": t,
            self.name + ".bk.t": t,
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

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._optic_edges()

        # DC fields at the mirror faces
        def get_fieldsDC(tp):
            try:
                return resultsDC[self.name + tp]
            except KeyError:
                return np.zeros((2, 1))

        fieldsDC_fr_i = get_fieldsDC(".fr.i.tp")
        fieldsDC_fr_o = get_fieldsDC(".fr.o.tp")
        fieldsDC_bk_i = get_fieldsDC(".bk.i.tp")
        fieldsDC_bk_o = get_fieldsDC(".bk.o.tp")

        # displacement to p (phase) quadrature
        px_fr = 4*np.pi/self.lambda_m * self.r * self.mlib.Mrotation(np.pi/2) @ fieldsDC_fr_i
        px_bk = 4*np.pi/self.lambda_m * self.r * self.mlib.Mrotation(np.pi/2) @ fieldsDC_bk_i

        # mechanical susceptibility
        chi = self.suscept_m_N(F_Hz).reshape((len(F_Hz), 1, 1))

        # q (amplitude) quadrature to displacement
        def xq_port(fieldsDC):
            return 2 / scc.c * chi * self.overlap * adjoint(fieldsDC)

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
