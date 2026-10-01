"""Interactive network drawings (PyVis / vis.js)."""

import html

import networkx as nx
import pandas as pd
from pyvis.network import Network

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#9a6324", "#469990", "#800000", "#808000", "#000075",
]
ROLE_SHAPES = {"Core": "star", "Broker": "diamond", "Inner circle": "triangle", "Member": "dot", "Peripheral": "dot"}


def community_color(cid) -> str:
    return PALETTE[int(cid) % len(PALETTE)] if cid is not None else "#999999"


def draw_network(
    G: nx.Graph,
    communities: dict,
    scores: pd.DataFrame,
    highlight: set | None = None,
    predicted_links: pd.DataFrame | None = None,
    height: str = "650px",
    new_edges: set | None = None,
) -> str:
    """Return an HTML page with the interactive network."""
    info = scores.set_index("person_id")
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    net.set_options("""{
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -35, "centralGravity": 0.02, "springLength": 60, "springConstant": 0.1, "avoidOverlap": 0.6},
        "stabilization": {"enabled": true, "iterations": 400, "fit": true}
      },
      "interaction": {"hover": true, "navigationButtons": true, "tooltipDelay": 120},
      "edges": {"smooth": false}
    }""")

    highlight = highlight or set()
    for n in G.nodes:
        row = info.loc[n] if n in info.index else None
        score = float(row["key_player_score"]) if row is not None else 0
        role = row["predicted_role"] if row is not None and "predicted_role" in row else "Member"
        attrs = G.nodes[n]
        tooltip = "\n".join([
            f"{attrs.get('name', n)} ({n})",
            f"Role: {role}",
            f"Key player score: {score:.1f}",
            f"Community: {communities.get(n, '-')}",
            f"City: {attrs.get('city', '-')}",
            f"Prior arrests: {attrs.get('prior_arrests', '-')}",
            f"Contacts: {G.degree(n)}",
        ])
        net.add_node(
            n,
            label=str(attrs.get("name", n)),
            title=html.escape(tooltip),
            color={"background": community_color(communities.get(n)),
                   "border": "#000000" if n in highlight else community_color(communities.get(n))},
            borderWidth=5 if n in highlight else 1,
            size=14 + score * 0.45,
            shape=ROLE_SHAPES.get(role, "dot"),
            font={"size": 16},
        )

    new_edges = new_edges or set()
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        is_new = tuple(sorted((u, v))) in new_edges
        net.add_edge(
            u, v,
            width=(4 if is_new else 1) + 5 * d["weight"] / max_w,
            # strong ties get short springs, so gangs cluster together on screen
            length=20 + 150 / (1 + d["weight"]),
            smooth=False,
            title=html.escape(f"Interactions: {d['interactions']}\nTypes: {', '.join(d['types'])}\nWeight: {d['weight']:.1f}"),
            color="#ff8c00" if is_new else "#b0b0b0",
        )

    if predicted_links is not None:
        for r in predicted_links.itertuples(index=False):
            if r.person_a in G and r.person_b in G:
                net.add_edge(
                    r.person_a, r.person_b,
                    dashes=True, width=4, color="#ff0000", physics=False, smooth=False,
                    title=f"PREDICTED hidden link ({r.probability:.0%})",
                )

    page = net.generate_html()
    # Freeze the layout once it has settled so the graph stops drifting, and fit it to the frame
    return page.replace(
        "return network;",
        'network.once("stabilizationIterationsDone", function () {'
        " network.setOptions({physics: false}); network.fit(); });\n return network;",
        1,
    )
