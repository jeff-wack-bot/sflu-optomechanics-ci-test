# Writing a model

This walks through the smallest useful optomechanical model: a Fabry–Perot
arm whose end mirror is free to move under radiation pressure. It is
`sflu.papers.klmtv2001.arm`, slightly simplified.

## 1. Topology

```python
from sflu import topologies

ifo = topologies.fp_arm()
```

`fp_arm` is a `GraphElement`: an ITM, an ETM with radiation pressure, and a
link `ARM.L` between them. It names its external nodes consistently:

* `ITM.bk.i.exc`: light (carrier, vacuum, squeezing) enters here;
* `ITM.bk.o.tp`: light leaves here, toward the readout;
* `ETM.pos.exc`: a displacement of the end mirror, which is where a GW
  signal enters.

To build a new topology, copy one of the builders in `sflu/topologies.py`:
`subgraph_add` the optics, then connect their ports with named edges.

## 2. Edges

The topology names edges, and edge objects supply their values:

```python
import numpy as np
from sflu import edges
from sflu.lib import MatrixLib

mlib = MatrixLib(nhom=0)              # fundamental mode only
M_kg, L_m, lambda_m = 7.5, 4e3, 1064e-9

edge_objs = [
    edges.MirrorEdge("ITM", Thr=0.014, mlib=mlib),
    edges.RPMirrorEdge("ETM", lambda_m=lambda_m, mlib=mlib,
                       suscept=lambda F: -1 / (M_kg * (2 * np.pi * F)**2)),
    edges.LinkEdge("ARM.L", L_m=L_m, mlib=mlib),
]
```

`RPMirrorEdge` needs the carrier field at the mirror to compute radiation
pressure, which is why there is a DC solve first.

## 3. Solve

```python
from sflu import solve

dc = solve.solve_dc(solve.build(topologies.fp_arm()), edge_objs, mlib,
                    drive={"ITM.bk.i.exc": mlib.LO(np.pi / 2)},
                    test_points={"ETM.fr.i.tp", "ETM.fr.o.tp"})
dc = solve.scale_dc(dc, "ETM.fr.i.tp", 750e3)    # 750 kW in the arm

F_Hz = np.geomspace(1, 1e4, 300)
T = solve.solve_ac(solve.build(topologies.fp_arm()), edge_objs, mlib, F_Hz,
                   readout="ITM.bk.o.tp",
                   inputs={"ITM.bk.i.exc", "ETM.pos.exc"},
                   resultsDC=dc)
```

`solve.build` reduces the graph in place, so build a fresh one for each solve.
`T` maps each input to its transfer matrix to the readout: `(N, 2, 2)` for an
optical input, and `(N, 2, 1)` for the mechanical one.

## 4. Read out

```python
from sflu import readout

row = readout.quadrature(mlib, np.pi / 2)        # phase readout
Sx, budget = readout.referred_psd(T, row, "ETM.pos.exc")
```

`Sx` is the quantum noise as a displacement PSD (single-sided, m²/Hz).
`budget` splits it by input port. To inject squeezing, describe the input's
state:

```python
states = {"ITM.bk.i.exc": readout.squeezed(mlib, dB=10, angle=-np.pi / 2)}
Sx_sqz, _ = readout.referred_psd(T, row, "ETM.pos.exc", states=states)
```

Angles may be arrays over frequency. That is how an ideal frequency-dependent
squeezer or readout is modelled; a real one is a filter cavity, and
`topologies.cascade` chains its transfer matrices onto the arm's.

## 5. Check it

Every model in `sflu.papers` comes with the paper's own closed-form result,
and its example asserts that the two agree. That is the standard to aim for,
because a model that only produces plausible plots is not yet known to be
right. Where there is no closed form, compare against an independent
calculation. `optics/test_simple_cavities.py`, for example, checks SFLU
against plain matrix algebra.
