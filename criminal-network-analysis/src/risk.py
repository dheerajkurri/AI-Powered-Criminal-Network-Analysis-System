"""Risk scoring, anomaly detection and (optional) supervised role classification."""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

NODE_FEATURES = [
    "degree", "strength", "betweenness", "closeness", "eigenvector", "pagerank",
    "clustering", "prior_arrests", "financial_links", "communities_linked",
]

RISK_LEVELS = [(75, "Critical"), (50, "High"), (25, "Medium"), (0, "Low")]


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def node_feature_table(G: nx.Graph, key_players: pd.DataFrame) -> pd.DataFrame:
    df = key_players.copy()
    clustering = nx.clustering(G, weight="weight")
    df["clustering"] = df["person_id"].map(clustering)
    df["prior_arrests"] = df["person_id"].map(lambda n: float(G.nodes[n].get("prior_arrests", 0) or 0))
    df["financial_links"] = df["person_id"].map(
        lambda n: sum(1 for m in G.neighbors(n) if "financial_transaction" in G[n][m]["types"])
    )
    return df


def risk_scores(features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Risk score (0-100) = network importance + criminal history + money flow.
    Anomaly flag  = Isolation Forest finds people whose contact pattern is
    unusual compared with everyone else (e.g. few contacts but many money links).
    """
    df = features.copy()
    risk = (
        0.30 * _minmax(df["pagerank"])
        + 0.20 * _minmax(df["betweenness"])
        + 0.25 * _minmax(df["prior_arrests"])
        + 0.15 * _minmax(df["financial_links"])
        + 0.10 * _minmax(df["communities_linked"])
    )
    df["risk_score"] = (100 * _minmax(risk)).round(1)
    df["risk_level"] = df["risk_score"].map(lambda s: next(label for t, label in RISK_LEVELS if s >= t))

    if len(df) >= 10:
        X = df[NODE_FEATURES].fillna(0).to_numpy()
        iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=seed)
        iso.fit(X)
        df["anomaly_score"] = (-iso.score_samples(X)).round(3)
        df["anomaly"] = iso.predict(X) == -1
    else:
        df["anomaly_score"] = 0.0
        df["anomaly"] = False
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def explain_anomaly(row: pd.Series, features: pd.DataFrame) -> str:
    """Plain-language reason: which features are far from the population average."""
    reasons = []
    for f in NODE_FEATURES:
        mean, std = features[f].mean(), features[f].std()
        if std and abs(row[f] - mean) / std >= 2:
            reasons.append(f"{'very high' if row[f] > mean else 'very low'} {f.replace('_', ' ')}")
    return ", ".join(reasons) or "unusual combination of features"


def train_role_classifier(features: pd.DataFrame, truth: pd.DataFrame, seed: int = 42) -> dict | None:
    """
    Supervised learning: when true roles are known for past cases, learn to
    predict a person's role (leader / lieutenant / broker / member / civilian)
    from their network features. Evaluated with 5-fold cross-validation.
    """
    df = features.merge(truth, on="person_id", how="inner")
    if df.empty or df["true_role"].value_counts().min() < 5:
        return None
    X = df[NODE_FEATURES].fillna(0)
    y = df["true_role"]
    model = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=cv)
    model.fit(X, y)
    report = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T
    return {
        "accuracy": round(accuracy_score(y, pred), 3),
        "macro_f1": round(f1_score(y, pred, average="macro"), 3),
        "report": report.round(3),
        "confusion": pd.crosstab(y, pred, rownames=["Actual"], colnames=["Predicted"]),
        "importance": pd.DataFrame({"feature": NODE_FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False),
        "predictions": pd.DataFrame({"person_id": df["person_id"], "true_role": y, "predicted_role": pred}),
    }
