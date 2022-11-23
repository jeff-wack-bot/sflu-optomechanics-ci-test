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

    def _edges(self, t, r):
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
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        edge_map.update({self.name + '.fr.px': self.mlib.Id})
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        resultsDC: the dictionary of DC results
        """
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        try:
            fieldsDC = resultsDC[self.name + '.fr.i.tp']
            rtW_m = 4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC
            edge_map.update({self.name + '.fr.px': rtW_m})
        except KeyError:
            pass
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

    def edgesAC(self, F_Hz):
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
            M_kg=None,
            lambda_m=1064e-9,
            mlib=MatrixLib(nhom=0),
    ):
        self.name = name
        self.t = np.sqrt(Thr)
        self.r = np.sqrt(1 - Thr)
        self.suscept_m_N = lambda F_Hz: -1/(M_kg * (2*np.pi*F_Hz)**2)
        self.lambda_m = lambda_m
        self.mlib = mlib

    def _optic_edges(self, r, t):
        edge_map = {
            self.name + ".fr.r": -r,
            self.name + ".bk.r": +r,
            self.name + ".fr.t": t,
            self.name + ".bk.t": t,
        }
        return edge_map

    def edgesDC(self):
        edge_map = self._optic_edges(self.mlib.diag(self.r), self.mlib.diag(self.t))
        # no radiation pressure at DC
        zz = {k: 0 * getattr(self.mlib, k) for k in ['Id_a', 'Id_v', 'Id_s']}
        edge_map.update({
            self.name + ".fr.Fq.i": zz["Id_a"],
            self.name + ".fr.Fq.o": zz["Id_a"],
            self.name + ".bk.Fq.i": zz["Id_a"],
            self.name + ".bk.Fq.o": zz["Id_a"],
            self.name + ".fr.px": zz["Id_v"],
            self.name + ".bk.px": zz["Id_v"],
            self.name + ".chi": zz["Id_s"],
        })
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._optic_edges(self.mlib.diag(self.r), self.mlib.diag(self.t))

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

        # q (amplitude) quadrature to force
        Fq_fr_i = +2/scc.c * adjoint(fieldsDC_fr_i)
        Fq_fr_o = +2/scc.c * adjoint(fieldsDC_fr_o)
        Fq_bk_i = -2/scc.c * adjoint(fieldsDC_bk_i)
        Fq_bk_o = -2/scc.c * adjoint(fieldsDC_bk_o)

        # mechanical susceptibility
        chi = self.suscept_m_N(F_Hz).reshape((len(F_Hz), 1, 1))

        edge_map.update({
            self.name + ".fr.Fq.i": Fq_fr_i,
            self.name + ".fr.Fq.o": Fq_fr_o,
            self.name + ".bk.Fq.i": Fq_bk_i,
            self.name + ".bk.Fq.o": Fq_bk_o,
            self.name + ".fr.px": px_fr,
            self.name + ".bk.px": px_bk,
            self.name + ".chi": chi,
        })
        return edge_map
