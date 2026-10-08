"""
Interferometers from the literature, rebuilt with SFLU.

Each module reproduces one paper. It holds three things, kept apart so that
one can check the others:

* the paper's **parameters**, with the equation or table they come from;
* the paper's **closed-form results**, transcribed from the paper;
* an **SFLU model** of the same interferometer, built from
  ``sflu.topologies`` and ``sflu.edges`` and solved with ``sflu.solve``.

The examples in ``papers/test_*.py`` plot the two together, reproducing the
paper's figures, and assert that they agree. Each module docstring states how
the paper's configuration was mapped onto the graph, and where a paper's
conventions differ from this package's (see ``sflu.readout`` for those).

=================  ==========================================================
module             paper
=================  ==========================================================
``klmtv2001``      Kimble, Levin, Matsko, Thorne, Vyatchanin, PRD 65 022002
``buonanno_chen2001``  Buonanno & Chen, PRD 64 042006 -- signal recycling
``buonanno_chen2002``  Buonanno & Chen, PRD 65 042001 -- optical springs
``buonanno_chen2003``  Buonanno & Chen, PRD 67 062002 -- scaling law
``harms2003``      Harms et al., PRD 68 042001 -- squeezed-input SR
``kwee2014``       Kwee et al., PRD 90 062006 -- filter-cavity decoherence
``purdue_chen2002``    Purdue & Chen, PRD 66 122004 -- speed meter
``korobko2019``    Korobko et al., LSA 8 118 -- quantum expander
=================  ==========================================================
"""
