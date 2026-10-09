"""Export Veritas belief graphs for visualization and analysis.

Formats:
- JSON: full fidelity, for programmatic use
- DOT: Graphviz visualization of justification dependencies
- GraphML: for Gephi/Cytoscape network analysis
"""

from __future__ import annotations

import json

from veritas.store import BeliefStore


def to_json(store: BeliefStore) -> str:
    """Export full store state as JSON."""
    return json.dumps({
        "beliefs": [
            {
                "id": b.id,
                "proposition": b.proposition,
                "confidence": b.confidence,
                "source": b.source,
                "source_reliability": b.source_reliability,
                "justifications": b.justifications,
                "status": b.status,
                "ground": b.ground,
                "timestamp": b.timestamp,
                "entrenchment": store.entrenchment_of(b),
            }
            for b in store.all_beliefs()
        ],
        "audit": [
            {
                "seq": e.seq,
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "belief_id": e.belief_id,
                "proposition": e.proposition,
                "reason": e.reason,
            }
            for e in store.audit
        ],
    }, indent=2)


def to_dot(store: BeliefStore) -> str:
    """Export justification graph as Graphviz DOT."""
    lines = ["digraph beliefs {", "  rankdir=BT;"]
    for b in store.all_beliefs():
        color = "green" if b.is_active else "red"
        shape = "box" if b.ground else "ellipse"
        label = b.proposition[:40].replace('"', "'")
        lines.append(
            f'  "{b.id}" [label="{label}", color={color}, shape={shape}];'
        )
        for jset in b.justifications:
            for pid in jset:
                lines.append(f'  "{pid}" -> "{b.id}";')
    lines.append("}")
    return "\n".join(lines)


def to_graphml(store: BeliefStore) -> str:
    """Export as GraphML for network analysis tools."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="proposition" for="node" attr.name="proposition" attr.type="string"/>',
        '  <key id="status" for="node" attr.name="status" attr.type="string"/>',
        '  <key id="entrenchment" for="node" attr.name="entrenchment" attr.type="double"/>',
        '  <graph id="beliefs" edgedefault="directed">',
    ]
    for b in store.all_beliefs():
        prop = b.proposition.replace("&", "&amp;").replace("<", "&lt;")
        lines.append(f'    <node id="{b.id}">')
        lines.append(f'      <data key="proposition">{prop}</data>')
        lines.append(f'      <data key="status">{b.status}</data>')
        lines.append(
            f'      <data key="entrenchment">{store.entrenchment_of(b):.4f}</data>'
        )
        lines.append('    </node>')
    eid = 0
    for b in store.all_beliefs():
        for jset in b.justifications:
            for pid in jset:
                lines.append(
                    f'    <edge id="e{eid}" source="{pid}" target="{b.id}"/>'
                )
                eid += 1
    lines.append('  </graph>')
    lines.append('</graphml>')
    return "\n".join(lines)
