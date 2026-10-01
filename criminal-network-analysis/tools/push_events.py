"""
Stand-in for an external data source (e.g. a telecom or bank system).

Appends new interaction records to data/live_interactions.csv every second.
In the dashboard, open the Live Monitor tab, choose "Watch CSV file" and press
Start; the dashboard picks up each new row as it is written.

    python tools/push_events.py            # runs until Ctrl+C
    python tools/push_events.py --rate 3   # 3 events per second
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import pipeline  # noqa: E402
from src.live import LIVE_COLUMNS, SimulatedFeed  # noqa: E402

DATA = os.path.join(ROOT, "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1, help="events per second")
    parser.add_argument("--out", default=os.path.join(DATA, "live_interactions.csv"))
    args = parser.parse_args()

    r = pipeline.run(pd.read_csv(os.path.join(DATA, "persons.csv")), pd.read_csv(os.path.join(DATA, "interactions.csv")))
    feed = SimulatedFeed(r["graph"], r["communities"], r["predicted_links"], r["scored"], seed=int(time.time()))

    with open(args.out, "w", newline="") as f:
        csv.DictWriter(f, LIVE_COLUMNS).writeheader()
    print(f"Writing live events to {args.out} (Ctrl+C to stop)")

    try:
        while True:
            events = feed.next_events(args.rate)
            with open(args.out, "a", newline="") as f:
                csv.DictWriter(f, LIVE_COLUMNS).writerows(events)
            for e in events:
                print(f"{e['date']}  {e['source']} -> {e['target']}  {e['interaction_type']} x{e['count']}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
