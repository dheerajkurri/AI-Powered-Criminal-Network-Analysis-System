"""Social network analysis (SNA) metrics used to identify key players."""

import networkx as nx
import pandas as pd

METRIC_LABELS = {
    "degree": "Degree (direct contacts)",
    "strength": "Strength (weighted contacts)",
    "betweenness": "Betweenness (bridge / broker)",
    "closeness": "Closeness (reach)",
    "eigenvector": "Eigenvector (connected to important people)",
    "pagerank": "PageRank (influence)",
}

# Weights of each metric in the combined "key player score".
SCORE_WEIGHTS = {
    "pagerank": 0.30,
    "betweenness": 0.25,
    "eigenvector": 0.20,
    "strength": 0.15,
    "closeness": 0.10,
}


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return series * 0
    return (series - lo) / (hi - lo)


def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """Return one row per person with all centrality metrics."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["person_id", "name", *METRIC_LABELS])

    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(G, distance="distance")
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=2000, tol=1e-6)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eigenvector = {n: 0.0 for n in G}
    pagerank = nx.pagerank(G, weight="weight")

    rows = []
    for n in G.nodes:
        rows.append({
            "person_id": n,
            "name": G.nodes[n].get("name", n),
            "degree": degree[n],
            "strength": round(strength[n], 2),
            "betweenness": betweenness[n],
            "closeness": closeness[n],
            "eigenvector": eigenvector[n],
            "pagerank": pagerank[n],
        })
    return pd.DataFrame(rows)


def rank_key_players(centrality: pd.DataFrame, communities: dict | None = None, G: nx.Graph | None = None) -> pd.DataFrame:
    """
    Combine normalised metrics into one 0-100 key player score and assign a
    structural position (rule based, from network shape only):
      Core          - top 5% influence (PageRank)
      Broker        - high betweenness and linked to several communities
      Inner circle  - top 20% influence
      Member        - regular participant
      Peripheral    - few, weak contacts
    """
    df = centrality.copy()
    if df.empty:
        return df
    score = sum(_minmax(df[m]) * w for m, w in SCORE_WEIGHTS.items())
    df["key_player_score"] = (100 * score / sum(SCORE_WEIGHTS.values())).round(1)

    if communities is not None and G is not None:
        df["community"] = df["person_id"].map(communities)
        df["communities_linked"] = df["person_id"].map(
            lambda n: len({communities[m] for m in G.neighbors(n)} - {communities[n]})
        )
    else:
        df["communities_linked"] = 0

    pr_hi = df["pagerank"].quantile(0.95)
    pr_mid = df["pagerank"].quantile(0.80)
    bt_hi = df["betweenness"].quantile(0.90)
    deg_lo = df["degree"].quantile(0.25)

    def role(r):
        if r["betweenness"] >= bt_hi and r["communities_linked"] >= 2:
            return "Broker"
        if r["pagerank"] >= pr_hi:
            return "Core"
        if r["pagerank"] >= pr_mid:
            return "Inner circle"
        if r["degree"] <= max(deg_lo, 1):
            return "Peripheral"
        return "Member"

    df["predicted_role"] = df.apply(role, axis=1)
    df = df.sort_values("key_player_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def ego_network(G: nx.Graph, node: str, radius: int = 1) -> nx.Graph:
    return nx.ego_graph(G, node, radius=radius)


def shortest_connection(G: nx.Graph, a: str, b: str) -> list[str] | None:
    """Strongest (shortest weighted) chain of contacts between two people."""
    try:
        return nx.shortest_path(G, a, b, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
