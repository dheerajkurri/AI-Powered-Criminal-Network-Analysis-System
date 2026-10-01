"""
Network disruption simulation.

Answers the investigator's question: "If we arrest these people, how badly
does the network break apart?" People are removed one by one according to a
strategy and we measure how much of the network is still connected.
"""

import random

import networkx as nx
import pandas as pd


def _state(G: nx.Graph, original_n: int) -> dict:
    if G.number_of_nodes() == 0:
        return {"largest_group_pct": 0.0, "groups": 0, "efficiency": 0.0}
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    return {
        "largest_group_pct": round(100 * largest / original_n, 2),
        "groups": nx.number_connected_components(G),
        "efficiency": round(nx.global_efficiency(G), 4),
    }


def simulate(G: nx.Graph, order: list[str], steps: int) -> pd.DataFrame:
    """Remove nodes in `order`, recording network health after each removal."""
    H = G.copy()
    n = G.number_of_nodes()
    rows = [{"removed": 0, "person_id": None, **_state(H, n)}]
    for i, node in enumerate(order[:steps], start=1):
        H.remove_node(node)
        rows.append({"removed": i, "person_id": node, **_state(H, n)})
    return pd.DataFrame(rows)


def compare_strategies(G: nx.Graph, ranked: pd.DataFrame, steps: int = 10, seed: int = 42) -> pd.DataFrame:
    """Run the simulation for several targeting strategies (plus a random baseline)."""
    strategies = {
        "Key player score": ranked.sort_values("key_player_score", ascending=False)["person_id"].tolist(),
        "Betweenness (brokers)": ranked.sort_values("betweenness", ascending=False)["person_id"].tolist(),
        "Degree (most contacts)": ranked.sort_values("degree", ascending=False)["person_id"].tolist(),
    }
    if "risk_score" in ranked.columns:
        strategies["Risk score"] = ranked.sort_values("risk_score", ascending=False)["person_id"].tolist()
    rnd = list(G.nodes)
    random.Random(seed).shuffle(rnd)
    strategies["Random arrests"] = rnd

    frames = []
    for name, order in strategies.items():
        df = simulate(G, order, steps)
        df["strategy"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
