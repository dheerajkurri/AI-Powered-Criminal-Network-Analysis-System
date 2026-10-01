"""
Live network monitoring.

New interactions arrive continuously from a source, the network is updated,
and an alert engine flags suspicious events as they happen.

Two sources are supported:
  SimulatedFeed - generates realistic events from the existing network (for demos)
  FileFeed      - watches a CSV file and picks up rows as they are appended
                  (how a real feed, e.g. call records exported by another
                  system, would be connected)
"""

import os
import random
from datetime import datetime

import pandas as pd

LIVE_COLUMNS = ["source", "target", "interaction_type", "date", "count"]
BIG_TRANSFER = 15  # financial_transaction count that triggers an alert


class SimulatedFeed:
    """Generates plausible new interactions based on the current network structure."""

    def __init__(self, G, communities: dict, predicted_links: pd.DataFrame, scored: pd.DataFrame, seed: int = 7):
        self.rng = random.Random(seed)
        self.G = G
        self.nodes = list(G.nodes)
        self.edges = list(G.edges)
        self.by_comm = {}
        for n, c in communities.items():
            self.by_comm.setdefault(c, []).append(n)
        self.comm = communities
        self.hidden = list(zip(predicted_links["person_a"], predicted_links["person_b"])) if not predicted_links.empty else []
        self.important = scored.sort_values("key_player_score", ascending=False)["person_id"].head(20).tolist()
        self.unknown_count = 0

    def _event(self, u, v, types, count_range):
        return {
            "source": u,
            "target": v,
            "interaction_type": self.rng.choice(types),
            "date": datetime.now().isoformat(timespec="seconds"),
            "count": self.rng.randint(*count_range),
        }

    def next_events(self, n: int) -> list[dict]:
        out = []
        for _ in range(n):
            r = self.rng.random()
            if r < 0.55 and self.edges:  # routine contact between people who already know each other
                u, v = self.rng.choice(self.edges)
                out.append(self._event(u, v, ["phone_call", "phone_call", "social_media", "meeting"], (1, 3)))
            elif r < 0.70:  # new contact inside the same group
                group = self.by_comm[self.rng.choice(list(self.by_comm))]
                if len(group) >= 2:
                    u, v = self.rng.sample(group, 2)
                    out.append(self._event(u, v, ["phone_call", "meeting"], (1, 4)))
            elif r < 0.80:  # contact between two different groups
                u = self.rng.choice(self.important)
                v = self.rng.choice([n for n in self.nodes if self.comm.get(n) != self.comm.get(u)])
                out.append(self._event(u, v, ["meeting", "phone_call", "financial_transaction"], (1, 5)))
            elif r < 0.88 and self.hidden:  # a link the AI predicted actually shows up
                u, v = self.rng.choice(self.hidden[:10])
                out.append(self._event(u, v, ["phone_call", "meeting"], (1, 3)))
            elif r < 0.95:  # a previously unknown person appears
                self.unknown_count += 1
                u = f"N{self.unknown_count:03d}"
                v = self.rng.choice(self.important if self.rng.random() < 0.6 else self.nodes)
                out.append(self._event(u, v, ["phone_call", "financial_transaction", "meeting"], (1, 4)))
            else:  # large money movement between key players
                u, v = self.rng.sample(self.important, 2)
                out.append(self._event(u, v, ["financial_transaction"], (BIG_TRANSFER, 30)))
        return out


class FileFeed:
    """Reads rows appended to a CSV file since the last read."""

    def __init__(self, path: str):
        self.path = path
        self.offset = 0

    def next_events(self, n: int | None = None) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            df = pd.read_csv(self.path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return []  # file is empty or being written right now; try again next tick
        if len(df) < self.offset:  # the file was restarted from scratch
            self.offset = 0
        new = df.iloc[self.offset:]
        if n:
            new = new.head(n)
        self.offset += len(new)
        for col, default in [("count", 1), ("date", datetime.now().isoformat(timespec="seconds"))]:
            if col not in new.columns:
                new = new.assign(**{col: default})
        return new[LIVE_COLUMNS].to_dict("records")


def add_event(G, event: dict) -> bool:
    """Add one interaction to the graph in place. Returns True if it created a new link."""
    from .graph_builder import DEFAULT_TYPE_WEIGHT, TYPE_WEIGHTS

    u, v = str(event["source"]), str(event["target"])
    if u == v:
        return False
    for n in (u, v):
        if n not in G:
            G.add_node(n, name=f"Unknown {n}", city=None, prior_arrests=0)
    w = int(event["count"]) * TYPE_WEIGHTS.get(event["interaction_type"], DEFAULT_TYPE_WEIGHT)
    if G.has_edge(u, v):
        d = G[u][v]
        d["weight"] += w
        d["interactions"] += int(event["count"])
        d["types"] = sorted(set(d["types"]) | {event["interaction_type"]})
        d["distance"] = 1.0 / d["weight"]
        return False
    G.add_edge(u, v, weight=w, interactions=int(event["count"]), types=[event["interaction_type"]], distance=1.0 / w)
    return True


def detect_alerts(event: dict, G, communities: dict, scored: pd.DataFrame, predicted: dict, known: set) -> list[dict]:
    """
    Check one incoming event against the current network and return alerts.
    `G` is the network *before* this event is added.
    """
    u, v = str(event["source"]), str(event["target"])
    info = scored.set_index("person_id")
    name = lambda n: G.nodes[n].get("name", n) if n in G else f"Unknown ({n})"
    level = lambda n: info.at[n, "risk_level"] if n in info.index else "Unknown"
    role = lambda n: info.at[n, "predicted_role"] if n in info.index else "Unknown"
    dangerous = lambda n: level(n) in ("High", "Critical") or role(n) in ("Core", "Broker")
    alerts = []

    def add(severity, kind, text):
        alerts.append({"time": event["date"], "severity": severity, "type": kind, "details": text,
                       "source": u, "target": v})

    for new, other in ((u, v), (v, u)):
        if new not in known:
            add("High" if dangerous(other) else "Medium", "Unknown person",
                f"Unidentified person {new} contacted {name(other)} ({level(other)} risk, {role(other)}) "
                f"via {event['interaction_type'].replace('_', ' ')}")

    pair = tuple(sorted((u, v)))
    if pair in predicted and not G.has_edge(u, v):
        add("High", "Prediction confirmed",
            f"AI-predicted hidden link {name(u)} ↔ {name(v)} (probability {predicted[pair]:.0%}) is now confirmed")

    if u in known and v in known and not G.has_edge(u, v) and communities.get(u) != communities.get(v):
        sev = "High" if dangerous(u) or dangerous(v) else "Low"
        add(sev, "Cross-group contact",
            f"First contact between {name(u)} (group {communities.get(u)}) and "
            f"{name(v)} (group {communities.get(v)})")

    if event["interaction_type"] == "financial_transaction" and int(event["count"]) >= BIG_TRANSFER:
        add("Critical" if dangerous(u) and dangerous(v) else "High", "Large money transfer",
            f"{event['count']} transactions between {name(u)} and {name(v)}")
    return alerts


def rank_changes(before: pd.DataFrame, after: pd.DataFrame, top: int = 10) -> list[str]:
    """People who entered the top-N key players since the last check."""
    b = set(before.nsmallest(top, "rank")["person_id"])
    a = after.nsmallest(top, "rank")
    return [f"{r.name} ({r.person_id}) entered the top {top} key players (now #{r.rank})"
            for r in a.itertuples() if r.person_id not in b]
