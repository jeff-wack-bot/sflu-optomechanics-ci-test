"""
Compatibility shim: the edge library now lives in ``sflu_components.edges``.

This module used to be a second copy of the edge classes, carrying features the
``sflu_components`` copy lacked (``SQZEdge``, ``edgesACSS`` state-space edges,
``MirrorEdge(loss_in_transmission=...)``) while lacking one the other had
(``loss_ports`` gating of the ``.fr.l`` / ``.bk.l`` edges). Stage 2b of
``REFACTOR_PLAN.md`` merged the union into ``sflu_components.edges``.

The merge was verified safe first: of the 80 edge-map entries both copies
produced, all 80 were bit-identical, so only the feature sets differed.

One behavioural note for anyone reading old code. This copy always emitted the
loss edges; the merged class emits them only when asked, matching the other
copy's callers. The models in this package therefore now pass
``loss_ports=True`` explicitly at each mirror. That is the same behaviour
written down instead of implied.

Import from ``sflu_components.edges`` in new code.
"""
from sflu_components.edges import (  # noqa: F401 - re-exported for compatibility
    BSEdge,
    LinkEdge,
    MirrorEdge,
    RPMirrorEdge,
    SQZEdge,
)

__all__ = [
    "BSEdge",
    "LinkEdge",
    "MirrorEdge",
    "RPMirrorEdge",
    "SQZEdge",
]
