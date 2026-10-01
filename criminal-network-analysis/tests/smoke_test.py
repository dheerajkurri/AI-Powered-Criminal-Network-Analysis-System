"""Runs the whole pipeline on the sample data and prints how accurate each AI module is."""

import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import communities, disruption, live, pipeline, risk  # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    persons = pd.read_csv(os.path.join(DATA, "persons.csv"))
    interactions = pd.read_csv(os.path.join(DATA, "interactions.csv"))
    truth = pd.read_csv(os.path.join(DATA, "ground_truth.csv"))

    t = time.time()
    r = pipeline.run(persons, interactions)
    print(f"Pipeline time: {time.time() - t:.1f}s")
    print("Summary:", r["summary"])
    print("Modularity:", round(r["modularity"], 3), "| communities:", len(set(r["communities"].values())))
    print("Community detection vs truth:", communities.evaluate_against_truth(r["communities"], truth))

    scored = r["scored"].merge(truth, on="person_id").sort_values("rank")
    print("\nTop 12 key players:")
    print(scored[["rank", "name", "key_player_score", "predicted_role", "true_role", "risk_level", "anomaly"]].head(12).to_string(index=False))
    print("\nRule-based role vs true role:")
    print(pd.crosstab(scored["true_role"], scored["predicted_role"]))

    lm = r["link_model"]
    print("\nLink prediction:", lm.metrics)
    print("Baseline AUCs:", lm.baseline_auc)
    print(r["predicted_links"].head(5).to_string(index=False))

    rc = risk.train_role_classifier(r["scored"], truth)
    print("\nRole classifier accuracy:", rc["accuracy"], "macro F1:", rc["macro_f1"])

    sim = disruption.compare_strategies(r["graph"], r["scored"], steps=10)
    print("\nLargest group % after 10 removals:")
    print(sim[sim["removed"] == 10][["strategy", "largest_group_pct", "groups"]].to_string(index=False))

    # Live monitoring: stream 200 simulated events through the alert engine
    G = r["graph"].copy()
    predicted = {tuple(sorted((a, b))): p for a, b, p in
                 zip(r["predicted_links"]["person_a"], r["predicted_links"]["person_b"], r["predicted_links"]["probability"])}
    feed = live.SimulatedFeed(G, r["communities"], r["predicted_links"], r["scored"])
    alerts, new_links = [], 0
    for e in feed.next_events(200):
        alerts += live.detect_alerts(e, G, r["communities"], r["scored"], predicted, set(G.nodes))
        new_links += live.add_event(G, e)
    print(f"\nLive: 200 events -> {new_links} new links, {len(alerts)} alerts:",
          pd.Series([a["type"] for a in alerts]).value_counts().to_dict())
    assert G.number_of_edges() == r["graph"].number_of_edges() + new_links

    assert r["summary"]["People"] == len(persons)
    assert lm.metrics["ROC-AUC"] > 0.7
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
