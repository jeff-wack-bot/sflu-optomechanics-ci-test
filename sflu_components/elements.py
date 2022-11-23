from wavestate.control.SFLU import optics


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
