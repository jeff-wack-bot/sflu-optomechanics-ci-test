"""
"""
import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
import scipy.constants as scc
from gwinc.struct import Struct
from gwinc.noise.quantum_lib import (
    mats_planewave,
    mats_mode_mismatch,
    adjoint,
    matrix_stack,
    Vnorm_sq,
)


pi2i = 2j*np.pi

mats_planewave.update(Struct(
    Id_s = np.ones((1, 1)),  # scalar identity
    Id_v = np.ones((2, 1)),  # vector identity
    Id_a = np.ones((1, 2)),  # adjoint identity
))


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
            mlib=mats_planewave,
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
            mlib=mats_planewave,
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


class HRMirrorRPReduced(optics.GraphElement):
    """
    GraphElement representing only the HR surface of a mirror but including
    radiation pressure effects. Gaussian elimination is done manually for the
    radiation pressure loop
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locations.update({
            'fr.i': (-4, +5),
            'fr.o': (-4, -5),
            'pos.exc': (+4, -5),
            'pos.tp': (+4, +5),
            'fr.F.i.exc': (+5, +8),
            'fr.i.tp': (-1, +8),
            'fr.o.tp': (-1, -8),
        })

        self.edges.update({
            ('fr.o', 'fr.i'): '.fr.r',
            ('pos.tp', 'fr.i'): '.fr.xq',
            ('fr.o', 'pos.exc'): '.fr.px',
            ('pos.tp', 'pos.exc'): '.xx',
            ('pos.tp', 'fr.F.i.exc'): '.fr.xF',
            ('fr.o', 'fr.F.i.exc'): '.fr.pF',
            ('fr.i.tp', 'fr.i'): '1',
            ('fr.o.tp', 'fr.o'): '1',
        })

    def properties(self, nodes, edges, rot_deg, **kwargs):
        nodes["fr.o"]['angle'] = +135
        nodes['fr.i']['angle'] = +135
        nodes['fr.F.i.exc']['angle'] = +45
        edges[('fr.o', 'fr.i')]['handed'] = 'r'
        edges[('fr.o', 'fr.F.i.exc')]['handed'] = 'r'


class HRMirrorRPReducedEdge:
    """
    Defines DC and AC edges for an HRMirrorRPReduced element representing only the
    HR surface of a mirror but including radiation pressure effects. Gaussian elimination
    is done manually for the radiation pressure loop.
    """
    def __init__(
            self,
            name,
            Thr=0,
            M_kg=None,
            lambda_m=1064e-9,
            mlib=mats_planewave,
    ):
        self.name = name
        self.t = np.sqrt(Thr)
        self.r = np.sqrt(1 - Thr)
        self.M_kg = M_kg
        self.lambda_m = lambda_m
        self.mlib = mlib

    def _optic_edges(self, r):
        edge_map = {
            self.name + '.fr.r': -r,
        }
        return edge_map

    def edgesDC(self):
        """
        Returns the DC edge map dictionary
        """
        edge_map = self._optic_edges(self.mlib.diag(self.r))
        # no radiation pressure at DC, and the matrix dimensions are correct
        zz = {k: np.zeros_like(self.mlib[k]) for k in self.mlib.keys() if 'Id' in k}
        edge_map.update({
            self.name + '.fr.xq': zz['Id_a'],
            self.name + '.fr.px': zz['Id_v'],
            self.name + '.xx': self.mlib['Id_s'],
            self.name + '.fr.pF': zz['Id'],
            self.name + '.fr.xF': zz['Id_a'],
        })
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        """
        Returns the AC edge map dictionary

        F_Hz: Frequency vector at which to evaluate the edge map
        resultsDC: the dictionary of DC results
        """
        edge_map = {}
        # DC fields at the front of the mirror
        fieldsDC_i = resultsDC[self.name + '.fr.i.tp']
        fieldsDC_o = resultsDC[self.name + '.fr.o.tp']
        # displacement to p (phase) quadrature [rtW/m]
        px = -4*np.pi/self.lambda_m * self.r * self.mlib.Mrotation(np.pi/2) @ fieldsDC_i
        # q (amplitude) quadrature to force [N/rtW]
        Fq_i = -2/scc.c * adjoint(fieldsDC_i)
        Fq_o = -2/scc.c * adjoint(fieldsDC_o)
        # mechanical susceptibility
        chi = -1/(self.M_kg * (2*np.pi*F_Hz)**2)
        chi = chi.reshape((len(F_Hz), 1, 1))  # is this necessary?
        # closed loop
        cl = self.mlib.Minv(self.mlib.Id - px @ chi @ Fq_o)
        fr_r = -self.mlib.diag(self.r) + px @ chi @ Fq_i
        # gotta be careful about signs when calling _optic_edges with full mirror...
        # edge_map['fr.r'] = self.mlib.diag(-self.r) + cl @ px @ chi @ fieldsDC_i
        edge_map = {
            'fr.r': cl @ fr_r,
            'fr.xq': chi @ (Fq_i + Fq_o @ cl @ fr_r),
            'fr.pF': cl @ px @ chi,
            'fr.px': cl @ px,
            'xx': self.mlib.Id_s + chi @ Fq_o @ cl @ px,
            'fr.xF': (self.mlib.Id_s + chi @ Fq_o @ cl @ px) @ chi,
        }
        edge_map = {self.name + '.' + k: v for k, v in edge_map.items()}
        return edge_map


class HRMirrorRP(optics.GraphElement):
    """
    GraphElement representing only the HR surface of a mirror but including
    radiation pressure effects in the full graph without any manual reduction
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locations.update({
            'fr.i': (-4, +5),
            'fr.o': (-4, -5),
            'fr.F.i': (5, +4),
            'fr.F.o': (5, -4),
            'pos': (5, 0),
            'fr.i.tp': (-1, +8),
            'fr.o.tp': (-1, -8),
        })

        self.edges.update({
            ('fr.o', 'fr.i'): '.fr.r',
            ('fr.F.i', 'fr.i'): '.fr.Fq.i',
            ('fr.F.o', 'fr.o'): '.fr.Fq.o',
            ('fr.o', 'pos'): '.px',
            ('pos', 'fr.F.i'): '.chi',
            ('pos', 'fr.F.o'): '.chi',
            ('fr.i.tp', 'fr.i'): '1',
            ('fr.o.tp', 'fr.o'): '1',
        })

        self.locations.update({
            'pos.exc': (+11, -2),
            'pos.tp': (+11, +2),
            'fr.F.i.exc': (+7, +8),
        })

        self.edges.update({
            ('pos.tp', 'pos'): '1s',
            ('pos', 'pos.exc'): '1s',
            ('fr.F.i', 'fr.F.i.exc'): '1a',
        })

    def properties(self, nodes, edges, rot_deg, **kwargs):
        nodes["fr.o"]['angle'] = +135
        nodes['fr.i']['angle'] = +135
        nodes['fr.F.i']['angle'] = -45
        nodes['pos']['angle'] = 150
        edges[('fr.o', 'pos')]['handed'] = 'l'
        edges[('pos', 'fr.F.o')]['handed'] = 'r'
        edges[('fr.F.o', 'fr.o')]['handed'] = 'l'
        edges[('fr.F.i', 'fr.i')]['handed'] = 'l'
        edges[('pos', 'fr.F.i')]['handed'] = 'r'
        edges[('pos.tp', 'pos')]['handed'] = 'r'
        edges[('pos', 'pos.exc')]['handed'] = 'r'


class HRMirrorRPEdge:
    def __init__(
            self,
            name,
            Thr=0,
            M_kg=None,
            lambda_m=1064e-9,
            mlib=mats_planewave,
    ):
        self.name = name
        self.t = np.sqrt(Thr)
        self.r = np.sqrt(1 - Thr)
        self.suscept_m_N = lambda F_Hz: -1/(M_kg * (2*np.pi*F_Hz)**2)
        self.lambda_m = lambda_m
        self.mlib = mlib

    def _optic_edges(self, r):
        edge_map = {
            self.name + '.fr.r': -r,
        }
        return edge_map

    def edgesDC(self):
        edge_map = self._optic_edges(self.mlib.diag(self.r))
        # no radiation pressure at DC
        zz = {k: np.zeros_like(self.mlib[k]) for k in self.mlib.keys() if 'Id' in k}
        edge_map.update({
            self.name + '.fr.Fq.i': zz['Id_a'],
            self.name + '.fr.Fq.o': zz['Id_a'],
            self.name + '.chi': zz['Id_s'],
            self.name + '.px': zz['Id_v'],
        })
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._optic_edges(self.mlib.diag(self.r))

        # DC fields at the front of the mirror
        fieldsDC_i = resultsDC[self.name + '.fr.i.tp']
        fieldsDC_o = resultsDC[self.name + '.fr.o.tp']

        # displacement to p (phase) quadrature
        px = -4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC_i

        # q (amplitude) quadrature to force
        Fq_i = -2/scc.c * adjoint(fieldsDC_i)
        Fq_o = -2/scc.c * adjoint(fieldsDC_o)

        # mechanical susceptibility
        # chi = -1/(self.M_kg * (2*np.pi*F_Hz)**2)
        chi = self.suscept_m_N(F_Hz).reshape((len(F_Hz), 1, 1))

        edge_map.update({
            self.name + '.fr.Fq.i': Fq_i,
            self.name + '.fr.Fq.o': Fq_o,
            self.name + '.px': px,
            self.name + '.chi': chi,
        })
        return edge_map


class RPMirrorElement(optics.GraphElement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locations.update({
            "fr.i": (-6, +7),
            "fr.o": (-6, -7),
            "bk.i": (+6, -7),
            "bk.o": (+6, +7),

            "fr.F.i": (-2, +5),
            "fr.F.o": (-2, -5),
            "bk.F.i": (+2, -5),
            "bk.F.o": (+2, +5),

            "pos": (0, 0),
        })
        self.locations.update({
            "fr.i.tp": (-9, +10),
            "fr.o.tp": (-9, -10),
            "bk.i.tp": (+9, -10),
            "bk.o.tp": (+9, +10),

            "pos.tp": (-9, 0),
            "pos.exc": (+9, 0),

            "fr.F.i.exc": (0, 10),
            "bk.F.i.exc": (0, -10),
        })

        self.edges.update({
            ("fr.o", "fr.i"): ".fr.r",
            ("bk.o", "bk.i"): ".bk.r",
            ("bk.o", "fr.i"): ".fr.t",
            ("fr.o", "bk.i"): ".bk.t",

            ("fr.F.i", "fr.i"): ".fr.Fq.i",
            ("fr.F.o", "fr.o"): ".fr.Fq.o",
            ("bk.F.i", "bk.i"): ".bk.Fq.i",
            ("bk.F.o", "bk.o"): ".bk.Fq.o",

            ("pos", "fr.F.i"): ".chi",
            ("pos", "fr.F.o"): ".chi",
            ("pos", "bk.F.i"): ".chi",
            ("pos", "bk.F.o"): ".chi",

            ("fr.o", "pos"): ".fr.px",
            ("bk.o", "pos"): ".bk.px",
        })
        self.edges.update({
            ("fr.i.tp", "fr.i"): "1",
            ("fr.o.tp", "fr.o"): "1",
            ("bk.i.tp", "bk.i"): "1",
            ("bk.o.tp", "bk.o"): "1",

            ("pos.tp", "pos"): "1s",
            ("pos", "pos.exc"): "1s",

            ("fr.F.i", "fr.F.i.exc"): "1a",
            ("bk.F.i", "bk.F.i.exc"): "1a",
        })

    def properties(self, nodes, edges, rot_deg, **kwargs):
        nodes["fr.i"]["angle"] = 135
        nodes["bk.o"]["angle"] = 45
        edges[("fr.o", "fr.i")]["handed"] = "r"
        super().properties(
            nodes=nodes,
            edges=edges,
            rot_deg=rot_deg,
            **kwargs,
        )
        return


class RPMirrorEdge:
    def __init__(
            self,
            name,
            Thr=0,
            M_kg=None,
            lambda_m=1064e-9,
            mlib=mats_planewave,
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
        zz = {k: np.zeros_like(self.mlib[k]) for k in self.mlib.keys() if 'Id' in k}
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
        px_fr = -4*np.pi/self.lambda_m * self.r * self.mlib.Mrotation(np.pi/2) @ fieldsDC_fr_i
        px_bk = -4*np.pi/self.lambda_m * self.r * self.mlib.Mrotation(np.pi/2) @ fieldsDC_bk_i

        # q (amplitude) quadrature to force
        Fq_fr_i = -2/scc.c * adjoint(fieldsDC_fr_i)
        Fq_fr_o = -2/scc.c * adjoint(fieldsDC_fr_o)
        Fq_bk_i = +2/scc.c * adjoint(fieldsDC_bk_i)
        Fq_bk_o = +2/scc.c * adjoint(fieldsDC_bk_o)

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
