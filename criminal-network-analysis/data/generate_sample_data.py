"""
Generates a synthetic (fictional) criminal network dataset for demonstration.

The network is built with a known hidden structure so the AI modules can be
evaluated against it:
  * several gangs, each with a leader, lieutenants and foot soldiers
  * a few "brokers" who connect different gangs
  * random noise contacts between unrelated people

Outputs (in this folder):
  persons.csv       - one row per person
  interactions.csv  - one row per recorded interaction between two people
  ground_truth.csv  - true gang / role of every person (for evaluation only)

All names and records are randomly generated and do not refer to real people.
"""

import os
import random
from datetime import date, timedelta

import pandas as pd

SEED = 42
N_GANGS = 6
GANG_SIZE_RANGE = (14, 24)
N_BROKERS = 5
N_ISOLATED_CIVILIANS = 12
NOISE_EDGES = 70

FIRST_NAMES = [
    "Arjun", "Ravi", "Imran", "Suresh", "Vikram", "Karthik", "Farhan", "Rahul",
    "Sanjay", "Naveen", "Ajay", "Salim", "Manoj", "Deepak", "Rohit", "Aman",
    "Kiran", "Priya", "Anjali", "Sneha", "Fatima", "Kavya", "Meera", "Pooja",
    "Ramesh", "Ganesh", "Abdul", "Harish", "Nikhil", "Tarun", "Varun", "Zaid",
    "Lakshmi", "Divya", "Ayesha", "Rekha", "Sunil", "Prakash", "Yusuf", "Mahesh",
]
LAST_NAMES = [
    "Reddy", "Sharma", "Khan", "Naidu", "Verma", "Rao", "Patel", "Singh",
    "Kumar", "Shaikh", "Gupta", "Iyer", "Das", "Joshi", "Mehta", "Ali",
    "Nair", "Chowdary", "Pillai", "Qureshi", "Mishra", "Yadav", "Goud", "Ansari",
]
CITIES = ["Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Pune", "Vijayawada", "Warangal"]
INTERACTION_TYPES = ["phone_call", "meeting", "financial_transaction", "co_arrest", "social_media"]


def random_date(rng, start=date(2023, 1, 1), days=900):
    return (start + timedelta(days=rng.randint(0, days))).isoformat()


def main():
    rng = random.Random(SEED)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    persons, truth, edges = [], [], []
    used_names = set()

    def new_person(gang, role, home_city):
        pid = f"P{len(persons) + 1:03d}"
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break
        prior = {"leader": rng.randint(3, 9), "lieutenant": rng.randint(2, 7),
                 "broker": rng.randint(1, 5), "member": rng.randint(0, 4),
                 "civilian": rng.choice([0, 0, 0, 1])}[role]
        persons.append({
            "person_id": pid,
            "name": name,
            "age": rng.randint(19, 58),
            "gender": rng.choice(["M"] * 4 + ["F"]),
            "city": home_city if rng.random() < 0.8 else rng.choice(CITIES),
            "prior_arrests": prior,
        })
        truth.append({"person_id": pid, "true_gang": gang, "true_role": role})
        return pid

    def add_edge(a, b, types, count_range):
        edges.append({
            "source": a,
            "target": b,
            "interaction_type": rng.choice(types),
            "date": random_date(rng),
            "count": rng.randint(*count_range),
        })

    gangs = {}
    for g in range(N_GANGS):
        gang_name = f"Gang-{chr(65 + g)}"
        city = CITIES[g % len(CITIES)]
        size = rng.randint(*GANG_SIZE_RANGE)
        leader = new_person(gang_name, "leader", city)
        lieutenants = [new_person(gang_name, "lieutenant", city) for _ in range(rng.randint(2, 3))]
        members = [new_person(gang_name, "member", city) for _ in range(size - 1 - len(lieutenants))]
        gangs[gang_name] = {"leader": leader, "lieutenants": lieutenants, "members": members}

        # Leader talks to lieutenants often, rarely to foot soldiers directly
        for lt in lieutenants:
            add_edge(leader, lt, ["meeting", "phone_call", "financial_transaction"], (8, 30))
        for lt_a in lieutenants:
            for lt_b in lieutenants:
                if lt_a < lt_b and rng.random() < 0.7:
                    add_edge(lt_a, lt_b, ["meeting", "phone_call"], (4, 15))
        # Each member reports to one or two lieutenants
        for m in members:
            for lt in rng.sample(lieutenants, k=rng.choice([1, 1, 2])):
                add_edge(lt, m, ["phone_call", "meeting", "financial_transaction"], (2, 12))
            if rng.random() < 0.12:
                add_edge(leader, m, ["phone_call"], (1, 3))
        # Members know each other inside the cell
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if rng.random() < 0.18:
                    add_edge(a, b, ["phone_call", "social_media", "co_arrest"], (1, 6))

    # Brokers connect 2-3 gangs through their leaders / lieutenants
    gang_names = list(gangs)
    for _ in range(N_BROKERS):
        broker = new_person("Unaffiliated", "broker", rng.choice(CITIES))
        for gname in rng.sample(gang_names, k=rng.choice([2, 3])):
            contact = rng.choice([gangs[gname]["leader"]] + gangs[gname]["lieutenants"])
            add_edge(broker, contact, ["financial_transaction", "meeting", "phone_call"], (3, 14))

    # A few people with minimal / unrelated contact
    all_ids = [p["person_id"] for p in persons]
    for _ in range(N_ISOLATED_CIVILIANS):
        civ = new_person("Unaffiliated", "civilian", rng.choice(CITIES))
        add_edge(civ, rng.choice(all_ids), ["phone_call", "social_media"], (1, 2))

    # Random noise between gangs
    all_ids = [p["person_id"] for p in persons]
    for _ in range(NOISE_EDGES):
        a, b = rng.sample(all_ids, 2)
        add_edge(a, b, ["phone_call", "social_media"], (1, 2))

    pd.DataFrame(persons).to_csv(os.path.join(out_dir, "persons.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
    pd.DataFrame(truth).to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)
    print(f"Generated {len(persons)} persons and {len(edges)} interactions in {out_dir}")


if __name__ == "__main__":
    main()
