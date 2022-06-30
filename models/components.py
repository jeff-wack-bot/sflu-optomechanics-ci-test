"""
"""
import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
import scipy.constants as scc
from gwinc.noise.quantum_lib import (
    mats_planewave,
    mats_mode_mismatch,
)


pi2i = 2j*np.pi

class GraphEdge:
    pass


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
        edge_map.update({self.name + '.X': self.mlib.Id})
        return edge_map

    def edgesAC(self, F_Hz, resultsDC):
        edge_map = self._edges(self.mlib.diag(self.t), self.mlib.diag(self.r))
        try:
            fieldsDC = resultsDC[self.name + '.fr.i.tp']
            rtW_m = 4*np.pi/self.lambda_m * self.mlib.Mrotation(np.pi/2) @ fieldsDC
            edge_map.update({self.name + '.X': rtW_m})
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
