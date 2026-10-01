"""Community (gang / cell) detection."""

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ALGORITHMS = {
    "Louvain": "louvain",
    "Greedy modularity": "greedy",
    "Label propagation": "label_propagation",
}


def detect_communities(G: nx.Graph, method: str = "louvain", seed: int = 42) -> tuple[dict, float]:
    """
    Split the network into tightly-knit groups.

    Returns (node -> community number, modularity). Communities are numbered
    by size, largest first. Modularity > 0.3 usually means a clear group structure.
    """
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes)}, 0.0

    if method == "louvain":
        groups = nx.community.louvain_communities(G, weight="weight", seed=seed)
    elif method == "greedy":
        groups = nx.community.greedy_modularity_communities(G, weight="weight")
    elif method == "label_propagation":
        groups = nx.community.asyn_lpa_communities(G, weight="weight", seed=seed)
    else:
        raise ValueError(f"Unknown community method: {method}")

    groups = sorted((set(g) for g in groups), key=len, reverse=True)
    modularity = nx.community.modularity(G, groups, weight="weight")
    mapping = {n: i for i, g in enumerate(groups) for n in g}
    return mapping, modularity


def community_summary(G: nx.Graph, communities: dict, key_players: pd.DataFrame) -> pd.DataFrame:
    """One row per community: size, density, most influential member, main city."""
    score = key_players.set_index("person_id")["key_player_score"]
    rows = []
    for cid in sorted(set(communities.values())):
        members = [n for n, c in communities.items() if c == cid]
        sub = G.subgraph(members)
        top = max(members, key=lambda n: score.get(n, 0))
        cities = pd.Series([G.nodes[n].get("city") for n in members]).dropna()
        external = sum(1 for n in members for m in G.neighbors(n) if communities[m] != cid)
        rows.append({
            "community": cid,
            "size": len(members),
            "internal_links": sub.number_of_edges(),
            "external_links": external,
            "density": round(nx.density(sub), 3),
            "likely_leader": f"{G.nodes[top].get('name', top)} ({top})",
            "leader_score": score.get(top, 0),
            "main_city": cities.mode().iat[0] if not cities.empty else "-",
            "avg_prior_arrests": round(pd.Series([G.nodes[n].get("prior_arrests", 0) for n in members]).mean(), 2),
        })
    return pd.DataFrame(rows)


def evaluate_against_truth(communities: dict, truth: pd.DataFrame) -> dict:
    """Compare detected communities with known gang labels (NMI and ARI, 1.0 = perfect)."""
    t = truth.dropna(subset=["true_gang"])
    t = t[~t["true_gang"].isin(["Unaffiliated", "None", ""])]
    t = t[t["person_id"].isin(communities)]
    if t.empty:
        return {}
    pred = [communities[p] for p in t["person_id"]]
    return {
        "NMI": round(normalized_mutual_info_score(t["true_gang"], pred), 3),
        "ARI": round(adjusted_rand_score(t["true_gang"], pred), 3),
        "people_evaluated": len(t),
    }
