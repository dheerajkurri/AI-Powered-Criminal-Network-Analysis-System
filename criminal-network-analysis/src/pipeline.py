"""Runs every analysis step in order. Used by the dashboard and the tests."""

from functools import partial

import pandas as pd

from . import analysis, communities, graph_builder, link_prediction, risk


def run(persons: pd.DataFrame, interactions: pd.DataFrame, *, types=None, start=None, end=None,
        min_weight: float = 0.0, community_method: str = "louvain", top_links: int = 25) -> dict:
    inter = graph_builder.filter_interactions(graph_builder.prepare_interactions(interactions), types, start, end)
    G = graph_builder.build_graph(persons, inter, min_weight=min_weight)

    comm, modularity = communities.detect_communities(G, community_method)
    centrality = analysis.compute_centrality(G)
    ranked = analysis.rank_key_players(centrality, comm, G)
    features = risk.node_feature_table(G, ranked)
    scored = risk.risk_scores(features)

    link_model = link_prediction.train_link_model(G, partial(communities.detect_communities, method=community_method))
    predicted = (link_prediction.predict_hidden_links(G, link_model, comm, top_k=top_links)
                 if link_model else pd.DataFrame())

    return {
        "graph": G,
        "interactions": inter,
        "summary": graph_builder.network_summary(G),
        "communities": comm,
        "modularity": modularity,
        "scored": scored,
        "community_summary": communities.community_summary(G, comm, ranked),
        "link_model": link_model,
        "predicted_links": predicted,
    }
