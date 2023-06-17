from wield.control.SFLU import optics


class MirrorElement(optics.GraphElement):
    def __init__(self, loss_ports=False, **kw):
        super().__init__(**kw)
        self._loss_ports = loss_ports
        self.locations.update({
            "fr.i": (-4, +5),
            "fr.o": (-4, -5),
            "bk.i": (+4, -5),
            "bk.o": (+4, +5),
        })
        self.edges.update({
            ("fr.o", "bk.i"): ".fr.t",
            ("bk.o", "fr.i"): ".fr.t",
            ("fr.o", "fr.i"): ".fr.r",
            ("bk.o", "bk.i"): ".bk.r",
        })
        if loss_ports:
            self.locations.update({
                "frL.i": (-2, -10),
                "bkL.i": (+2, +10),
            })
            self.edges.update({
                ("fr.o", "frL.i"): ".fr.l",
                ("bk.o", "bkL.i"): ".bk.l",
            })

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            nodes["fr.o"]['angle'] = +45
            nodes["bk.i"]['angle'] = +45
            if self._loss_ports:
                edges[("fr.o", "frL.i")]['handed'] = 'r'
                edges[("fr.o", "frL.i")]['dist'] = 0.2

                nodes["bkL.i"]['angle'] = +45
            pass
        elif rot_deg < 135:
            edges[("fr.o", "fr.i")]['handed'] = 'r'
            edges[("bk.o", "bk.i")]['handed'] = 'r'
            nodes["fr.i"]['angle'] = +45
            nodes["fr.o"]['angle'] = +45
            if self._loss_ports:
                edges[("fr.o", "frL.i")]['dist'] = 0.2
                nodes["frL.i"]['angle'] = +45
                nodes["bkL.i"]['angle'] = -135
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            # edges[("fr.o", "bk.i")]['handed'] = 'r'
            # edges[("bk.o", "fr.i")]['handed'] = 'r'
            nodes["bk.o"]['angle'] = +45
            nodes["fr.i"]['angle'] = +45
            if self._loss_ports:
                edges[("bk.o", "bkL.i")]['handed'] = 'r'
                edges[("bk.o", "bkL.i")]['dist'] = 0.2

                nodes["frL.i"]['angle'] = +45
            pass
        elif rot_deg < 270 + 45:
            edges[("fr.o", "fr.i")]['handed'] = 'r'
            edges[("bk.o", "bk.i")]['handed'] = 'r'
            nodes["bk.i"]['angle'] = +45
            nodes["bk.o"]['angle'] = +45
            if self._loss_ports:
                nodes["frL.i"]['angle'] = -135
                nodes["bkL.i"]['angle'] = +45
                edges[("bk.o", "bkL.i")]['dist'] = 0.2
            # ~270deg
            pass
        else:
            pass
        super().properties(
            nodes=nodes,
            edges=edges,
            rot_deg=rot_deg,
            **kw
        )
        return


class BeamSplitterElement(optics.GraphElement):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.locations.update({
            "frA.i": (-10, +5),
            "frA.o": (-10, -5),
            "bkA.i": (+10, -5),
            "bkA.o": (+10, +5),
            "frB.o": (-5, +10),
            "frB.i": (+5, +10),
            "bkB.i": (-5, -10),
            "bkB.o": (+5, -10),
        })
        self.edges.update({
            ("bkA.o", "frA.i"): ".t",
            ("frA.o", "bkA.i"): ".t",
            ("bkB.o", "frB.i"): ".t",
            ("frB.o", "bkB.i"): ".t",
            ("frB.o", "frA.i"): ".fr.r",
            ("frA.o", "frB.i"): ".fr.r",
            ("bkA.o", "bkB.i"): ".bk.r",
            ("bkB.o", "bkA.i"): ".bk.r",
        })

    def properties(self, nodes, edges, rot_deg, **kw):
        if rot_deg < 45:
            # ~0deg
            pass
        elif rot_deg < 135:
            # ~90deg
            pass
        elif rot_deg < 180 + 45:
            # ~180deg
            pass
        elif rot_deg < 270 + 45:
            # ~270deg
            pass
        else:
            pass
        super().properties(
            nodes=nodes,
            edges=edges,
            rot_deg=rot_deg,
            **kw
        )
        return


class RPMirrorElement(optics.GraphElement):
    def __init__(self, loss_ports=False,
                 **kwargs):
        super().__init__(**kwargs)
        self._loss_ports = loss_ports
        self.locations.update({
            "fr.i": (-6, +7),
            "fr.o": (-6, -7),
            "bk.i": (+6, -7),
            "bk.o": (+6, +7),

            "pos": (0, 0),

            "fr.i.tp": (-9, +10),
            "fr.o.tp": (-9, -10),
            "bk.i.tp": (+9, -10),
            "bk.o.tp": (+9, +10),

            "pos.tp": (-9, 0),
            "pos.exc": (+9, 0),
        })
        self.edges.update({
            ("fr.o", "fr.i"): ".fr.r",
            ("bk.o", "bk.i"): ".bk.r",
            ("bk.o", "fr.i"): ".fr.t",
            ("fr.o", "bk.i"): ".bk.t",

            ("pos", "fr.i"): ".fr.xq.i",
            ("pos", "fr.o"): ".fr.xq.o",
            ("pos", "bk.i"): ".bk.xq.i",
            ("pos", "bk.o"): ".bk.xq.o",

            ("fr.o", "pos"): ".fr.px",
            ("bk.o", "pos"): ".bk.px",

            ("fr.i.tp", "fr.i"): "1",
            ("fr.o.tp", "fr.o"): "1",
            ("bk.i.tp", "bk.i"): "1",
            ("bk.o.tp", "bk.o"): "1",

            ("pos.tp", "pos"): "1s",
            ("pos", "pos.exc"): "1s",

        })
        if loss_ports:
            self.locations.update({
                "frL.i": (-3, -10),
                "bkL.i": (+4, +10),
            })
            self.edges.update({
                ("fr.o", "frL.i"): ".fr.l",
                ("bk.o", "bkL.i"): ".bk.l",
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
