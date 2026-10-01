"""
Hidden link prediction.

Police records are always incomplete. This module learns what a real link
looks like from the known network and then scores every unconnected pair of
people to find likely hidden relationships.

Training procedure (standard for link prediction):
  1. Hide a random 20% of the known links.
  2. Compute topological features on the remaining graph.
  3. Train a Random Forest to separate hidden links (positives) from pairs
     that are truly unconnected (negatives).
  4. Measure accuracy on a held-out test split (ROC-AUC, precision, recall).
  5. Re-compute features on the full graph and score all unconnected pairs.
"""

import math
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
    "same_community",
    "shortest_path",
    "same_city",
    "prior_arrests_sum",
]


@dataclass
class LinkModel:
    model: RandomForestClassifier
    metrics: dict
    baseline_auc: dict
    feature_importance: pd.DataFrame


def _features(G: nx.Graph, pairs: list[tuple], communities: dict) -> pd.DataFrame:
    rows = []
    deg = dict(G.degree())
    for u, v in pairs:
        cn = list(nx.common_neighbors(G, u, v))
        union = len(set(G[u]) | set(G[v]))
        try:
            sp = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            sp = 10
        rows.append({
            "u": u,
            "v": v,
            "common_neighbors": len(cn),
            "jaccard": len(cn) / union if union else 0.0,
            "adamic_adar": sum(1 / math.log(deg[w]) for w in cn if deg[w] > 1),
            "resource_allocation": sum(1 / deg[w] for w in cn if deg[w] > 0),
            "preferential_attachment": deg[u] * deg[v],
            "same_community": int(communities.get(u) == communities.get(v)),
            "shortest_path": min(sp, 10),
            "same_city": int(G.nodes[u].get("city") is not None and G.nodes[u].get("city") == G.nodes[v].get("city")),
            "prior_arrests_sum": float(G.nodes[u].get("prior_arrests", 0) or 0) + float(G.nodes[v].get("prior_arrests", 0) or 0),
        })
    return pd.DataFrame(rows)


def _sample_non_edges(G: nx.Graph, k: int, rng: random.Random, max_distance: int = 3) -> list[tuple]:
    """
    Half the negatives are random pairs, half are "near misses" (2-3 hops apart
    but not linked). Near misses are the hard cases the model will face when
    scoring candidates, so without them it only learns "same gang = linked".
    """
    nodes = list(G.nodes)
    chosen = set()
    attempts = 0
    while len(chosen) < k // 2 and attempts < k * 50:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        if not G.has_edge(u, v):
            chosen.add(tuple(sorted((u, v))))
    near = sorted({
        tuple(sorted((u, v)))
        for u in nodes
        for v, d in nx.single_source_shortest_path_length(G, u, cutoff=max_distance).items()
        if d >= 2
    } - chosen)
    chosen.update(rng.sample(near, min(k - len(chosen), len(near))))
    return sorted(chosen)  # sorted so results are identical on every run


def train_link_model(G: nx.Graph, community_fn, hide_fraction: float = 0.2, seed: int = 42) -> LinkModel | None:
    """Train and evaluate the link prediction model. Returns None if the graph is too small."""
    rng = random.Random(seed)
    edges = list(G.edges())
    if len(edges) < 30:
        return None

    hidden = rng.sample(edges, int(len(edges) * hide_fraction))
    G_train = G.copy()
    G_train.remove_edges_from(hidden)
    communities, _ = community_fn(G_train)

    negatives = _sample_non_edges(G, 2 * len(hidden), rng)
    X = pd.concat([_features(G_train, hidden, communities), _features(G_train, negatives, communities)], ignore_index=True)
    y = np.array([1] * len(hidden) + [0] * len(negatives))

    X_train, X_test, y_train, y_test = train_test_split(X[FEATURES], y, test_size=0.3, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=seed, class_weight="balanced")
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "ROC-AUC": round(roc_auc_score(y_test, prob), 3),
        "Average precision": round(average_precision_score(y_test, prob), 3),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 3),
        "Training pairs": len(X_train),
        "Test pairs": len(X_test),
    }
    # Classic single-heuristic scores, to show the ML model beats them
    baseline = {f: round(roc_auc_score(y_test, X_test[f]), 3) for f in ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"]}
    importance = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return LinkModel(model, metrics, baseline, importance)


def predict_hidden_links(G: nx.Graph, link_model: LinkModel, communities: dict, top_k: int = 25, max_distance: int = 3) -> pd.DataFrame:
    """
    Score unconnected pairs that are within `max_distance` hops of each other
    (people further apart are very unlikely to be secretly linked) and return
    the most probable hidden links.
    """
    candidates = set()
    for u in G.nodes:
        for v, d in nx.single_source_shortest_path_length(G, u, cutoff=max_distance).items():
            if d >= 2 and u < v:
                candidates.add((u, v))
    if not candidates:
        return pd.DataFrame(columns=["person_a", "person_b", "probability"])

    feats = _features(G, sorted(candidates), communities)
    feats["probability"] = link_model.model.predict_proba(feats[FEATURES])[:, 1]
    feats = feats.sort_values("probability", ascending=False).head(top_k)
    name = lambda n: G.nodes[n].get("name", n)
    return pd.DataFrame({
        "person_a": feats["u"],
        "name_a": feats["u"].map(name),
        "person_b": feats["v"],
        "name_b": feats["v"].map(name),
        "probability": feats["probability"].round(3),
        "common_contacts": feats["common_neighbors"],
        "same_community": feats["same_community"].map({1: "Yes", 0: "No"}),
    }).reset_index(drop=True)
