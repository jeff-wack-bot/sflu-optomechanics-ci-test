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
)


pi2i = 2j*np.pi

mats_planewave.update(Struct(
    Id_s = np.ones((1, 1)),
    Id_v = np.ones((2, 1)),
    Id_a = np.ones((1, 2)),
))


def assert_optic_properties(R_or_T, L=0):
    assert(R_or_T >= 0 and R_or_T <= 1)
    assert(L >= 0 and L <= 1)
    T_or_R = 1 - (R_or_T + L)
    assert(T_or_R >= 0 and T_or_R <= 1)


class MirrorEdge:
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
        assert_optic_properties(Thr, Lhr)
        assert_optic_properties(Rar)

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
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        edge_map.update({self.name + '.fr.px': self.mlib.Id})
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        try:
            fieldsDC = resultsDC[self.name + '.fr.i.tp']
            rtW_m = 4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC
            edge_map.update({self.name + '.fr.px': rtW_m})
        except KeyError:
            pass
        return edge_map


class MirrorEdgeRP:
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
        assert_optic_properties(Thr, Lhr)
        assert_optic_properties(Rar)

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
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        dim = self.mlib.Id.shape[0]
        zz = np.zeros_like(self.mlib.Id)

        # edge_map.update({self.name + '.fr.Fq': zz})
        # edge_map.update({self.name + '.fr.chi': zz})
        # edge_map.update({self.name + '.fr.px': zz})

        edge_map.update({self.name + '.fr.Fq': np.zeros((1, 2))})
        edge_map.update({self.name + '.fr.chi': np.zeros((1, 1))})
        edge_map.update({self.name + '.fr.px': np.zeros((2, 1))})

        # edge_map.update({self.name + '.fr.px': self.mlib.Id})
        # edge_map.update({self.name + '.fr.Fq': self.mlib.Id})
        # edge_map.update({self.name + '.fr.chi': self.mlib.Id})
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        try:
            fieldsDC = resultsDC[self.name + '.fr.i.tp']
            rtW_m = 4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC
            edge_map[self.name + '.fr.px'] = rtW_m
            # edge_map.update({self.name + '.X': rtW_m})

            edge_map[self.name + '.fr.Fq'] = 4/scc.c * adjoint(fieldsDC)
            edge_map[self.name + '.fr.chi'] = np.ones((len(F_Hz), 1, 1))
        except KeyError:
            pass
        return edge_map


class LinkEdge:
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
        Lmat = self.mlib.Mrotation(self.detune_rad)
        return self._edges(Lmat)

    def edgesAC(self, F_Hz):
        delay = self.mlib.diag(np.exp(-pi2i * F_Hz * self.L_m / scc.c))
        Lmat = delay @ self.mlib.Mrotation(self.detune_rad)
        return self._edges(Lmat)


class MirrorRP(optics.Mirror):
    """
    A mirror that can have a mechanical susceptibility and experience radiation pressure
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locations.update({
            "fr.i": (-6, +8),
            "fr.o": (-6, -8),
            "bk.i": (+6, -8),
            "bk.o": (+6, +8),
        })
        self.locations.update({
            "fr.F": (0, 3),
            "pos":  (0, 0),

            # "fr.a": (-4, +6),
            "fr.v": (-4, -6),
        })
        self.locations.update({
            "pos.exc": (+9, 0),
            "pos.tp":  (-9, 0),
            "fr.F.exc": (+3, +11),
        })

        self.edges.update({
            ("bk.o", "fr.i"): ".fr.t",
            ("fr.o", "bk.i"): ".bk.t",
            ("fr.o", "fr.i"): ".fr.r",
            ("bk.o", "bk.i"): ".bk.r",
        })
        self.edges.update({
            # ("fr.a", "fr.i"): "1a",
            # ("fr.F", "fr.a"): ".fr.Fq",
            ("fr.F", "fr.i"): ".fr.Fq",  # q quadrature (amplitude) field to force
            ("pos", "fr.F"): ".fr.chi",  # mechanical susceptibility
            # ("fr.o", "pos"): ".fr.px",   # position to p quadrature (phase) field
            ("fr.v", "pos"): ".fr.px",
            ("fr.o", "fr.v"): "1v",

            # ("fr.a", "fr.i"): ".1a",
            # ("fr.F", "fr.a"): ".fr.Fq",  # q quadrature (amplitude) field to force
            # ("pos", "fr.F"): ".fr.chi",  # mechanical susceptibility
            # ("fr.v", "pos"): ".fr.px",   # position to p quadrature (phase) field
            # ("fr.o", "fr.v"): ".1v",
        })
        self.edges.update({
            ("pos.tp", "pos"): "1s",
            ("pos", "pos.exc"): "1s",
            ("fr.F", "fr.F.exc"): "1a",
        })

        #     "fr.F": (0, 3),
        #     "pos":  (0, 0),

        #     "pos.exc": (+9, 0),
        #     "pos.tp":  (-9, 0),
        #     "fr.F.exc": (+3, +11),
        # })
        # self.edges.update({
        #     ("bk.o", "fr.i"): ".fr.t",
        #     ("fr.o", "bk.i"): ".bk.t",
        #     ("fr.o", "fr.i"): ".fr.r",
        #     ("bk.o", "bk.i"): ".bk.r",

        #     ("fr.F", "fr.i"): ".fr.Fq",  # q quadrature (amplitude) field to force
        #     ("pos", "fr.F"): ".fr.chi",  # mechanical susceptibility
        #     ("fr.o", "pos"): ".fr.px",   # position to p quadrature (phase) field

        #     ("pos.tp", "pos"): "1s",
        #     ("pos", "pos.exc"): "1s",
        #     ("fr.F", "fr.F.exc"): "1a",
        # })

    def properties(self, nodes, edges, rot_deg, **kwargs):
        nodes["fr.F"]['angle'] = +45
        nodes["fr.i"]['angle'] = +135
        nodes["fr.o"]['angle'] = -135
        nodes["pos.tp"]['angle'] = -135
        edges[("fr.F", "fr.i")]['handed'] = 'l'
        super().properties(
            nodes=nodes,
            edges=edges,
            rot_deg=rot_deg,
            **kwargs,
        )


class SimpleMirror(optics.GraphElement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.locations.update({
            "fr.i": (-6, +8),
            "fr.o": (-6, -8),
            "pos": (0, 0),
            "fr.F": (0, 5),
        })
        # self.locations.update({
        #     "fr.a": (-4, +6),
        #     "fr.v": (-4, -6),
        # })
        self.locations.update({
            "Fr.i.exc": (-12, +8),
            "fr.o.exc": (-5, -11),
            "fr.i.tp": (-5, +11),
            "fr.o.tp": (-12, -8),
            "fr.F.exc": (3, +13),
            "pos.tp": (-12, 0),
            "pos.exc": (7, 0),
        })
        self.edges.update({
            ("fr.o", "fr.i"): ".fr.r",
            ("fr.F", "fr.i"): ".fr.Fq",
            ("pos", "fr.F"): ".fr.chi",
            ("fr.o", "pos"): ".fr.px",
        })
        # self.edges.update({
        #     ("fr.o", "fr.i"): ".fr.r",
        #     ("fr.a", "fr.i"): '1a',
        #     ("fr.F", "fr.a"): ".fr.Fq",
        #     ("pos", "fr.F"): ".fr.chi",
        #     ("fr.v", "pos"): ".fr.px",
        #     ("fr.o", "fr.v"): "1v",
        # })
        self.edges.update({
            ("fr.i", "fr.i.exc"): "1",
            ("fr.i.tp", "fr.i"): "1",
            ("fr.o.tp", "fr.o"): "1",
            ("fr.o", "fr.o.exc"): "1",

            ("fr.F", "fr.F.exc"): "1a",
            ("pos", "pos.exc"): "1s",
            ("pos.tp", "pos"): "1s",
        })

    def properties(self, nodes, edges, rot_deg, **kwargs):
        nodes["fr.F"]['angle'] = +45
        nodes["fr.i"]['angle'] = +135
        nodes["fr.o.tp"]['angle'] = +135
        nodes["fr.i.exc"]['angle'] = -135


class SimpleMirrorEdge:
    def __init__(self,
                 name,
                 Rhr=1,
                 lambda_m=1064e-9,
                 mlib=mats_planewave,
    ):
        self.name = name
        self.r = np.sqrt(Rhr)
        self.lambda_m = lambda_m
        self.mlib = mlib

    def _edges(self, r):
        edge_map = {
            self.name + '.fr.r': -r,
        }
        return edge_map

    def edgesDC(self):
        edge_map = self._edges(self.mlib.diag(self.r))
        edge_map[self.name + '.fr.Fq'] = np.zeros_like(self.mlib.Id_a)
        edge_map[self.name + '.fr.chi'] = np.zeros_like(self.mlib.Id_s)
        edge_map[self.name + '.fr.px'] = np.zeros_like(self.mlib.Id_v)
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        npts = len(F_Hz)
        edge_map = self._edges(self.mlib.diag(self.r))
        fieldsDC = resultsDC[self.name + '.fr.i.tp']
        # q (amplitude) quadrature to force [N/rtW]
        Fq = 4/scc.c * adjoint(fieldsDC)
        Fq = np.repeat(Fq[np.newaxis, :, :], npts, axis=0)
        edge_map[self.name + '.fr.Fq'] = Fq
        # mechanical susceptibility [m/N]
        edge_map[self.name + '.fr.chi'] = np.ones((len(F_Hz), 1, 1))
        # displacement to p (phase) quadrature [rtW/m]
        px = 4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC
        px = np.repeat(px[np.newaxis, :, :], npts, axis=0)
        edge_map[self.name + '.fr.px'] = px
        return edge_map


class HRMirrorRPReduced(optics.GraphElement):
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
        edge_map = self._optic_edges(self.mlib.diag(self.r))
        zz = np.zeros_like(self.mlib.Id)
        edge_map.update({
            self.name + '.fr.xq': zz,
            self.name + '.fr.px': zz,
            self.name + '.xx': zz,
            self.name + '.fr.pF': zz,
            self.name + '.fr.xF': zz,
        })
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = {}
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
