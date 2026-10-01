"""Loading, validating and converting raw records into a weighted network."""

import networkx as nx
import pandas as pd

PERSON_COLUMNS = {"person_id", "name"}
INTERACTION_COLUMNS = {"source", "target", "interaction_type"}

# How strongly each kind of interaction indicates a real criminal tie.
# Money and face-to-face meetings are stronger evidence than a social media contact.
TYPE_WEIGHTS = {
    "financial_transaction": 3.0,
    "meeting": 2.5,
    "co_arrest": 2.0,
    "phone_call": 1.0,
    "social_media": 0.5,
}
DEFAULT_TYPE_WEIGHT = 1.0


def validate(persons: pd.DataFrame, interactions: pd.DataFrame) -> list[str]:
    """Return a list of problems with the uploaded data (empty list = valid)."""
    problems = []
    missing = PERSON_COLUMNS - set(persons.columns)
    if missing:
        problems.append(f"persons file is missing columns: {sorted(missing)}")
    missing = INTERACTION_COLUMNS - set(interactions.columns)
    if missing:
        problems.append(f"interactions file is missing columns: {sorted(missing)}")
    if not problems and persons["person_id"].duplicated().any():
        problems.append("persons file has duplicate person_id values")
    return problems


def prepare_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Fill optional columns and normalise types."""
    df = interactions.copy()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["interaction_type"] = df["interaction_type"].astype(str).str.strip().str.lower()
    if "count" not in df.columns:
        df["count"] = 1
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).clip(lower=1)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    return df[df["source"] != df["target"]]


def filter_interactions(df, types=None, start=None, end=None):
    """Keep only the selected interaction types and date window."""
    out = df
    if types:
        out = out[out["interaction_type"].isin(types)]
    if start is not None and out["date"].notna().any():
        out = out[out["date"].isna() | (out["date"] >= pd.Timestamp(start))]
    if end is not None and out["date"].notna().any():
        out = out[out["date"].isna() | (out["date"] <= pd.Timestamp(end))]
    return out


def build_graph(persons: pd.DataFrame, interactions: pd.DataFrame, min_weight: float = 0.0) -> nx.Graph:
    """
    Build an undirected weighted graph.

    Several interactions between the same pair are merged into one edge whose
    weight is sum(count * type_weight). `distance` = 1 / weight is stored for
    shortest-path based metrics (strong tie = short distance).
    """
    G = nx.Graph()
    people = persons.copy()
    people["person_id"] = people["person_id"].astype(str).str.strip()
    for row in people.to_dict("records"):
        G.add_node(row["person_id"], **{k: v for k, v in row.items() if k != "person_id" and pd.notna(v)})

    for row in interactions.itertuples(index=False):
        u, v = row.source, row.target
        w = row.count * TYPE_WEIGHTS.get(row.interaction_type, DEFAULT_TYPE_WEIGHT)
        if G.has_edge(u, v):
            data = G[u][v]
            data["weight"] += w
            data["interactions"] += int(row.count)
            data["types"].add(row.interaction_type)
        else:
            G.add_edge(u, v, weight=w, interactions=int(row.count), types={row.interaction_type})

    for node in G.nodes:
        G.nodes[node].setdefault("name", node)

    weak = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_weight]
    G.remove_edges_from(weak)

    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / d["weight"]
        d["types"] = sorted(d["types"])
    return G


def network_summary(G: nx.Graph) -> dict:
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    return {
        "People": G.number_of_nodes(),
        "Links": G.number_of_edges(),
        "Density": round(nx.density(G), 4),
        "Connected groups": len(components),
        "Largest group size": len(largest),
        "Average connections": round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2),
        "Clustering coefficient": round(nx.average_clustering(G, weight="weight"), 4),
        "Isolated people": nx.number_of_isolates(G),
    }
