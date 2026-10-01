# AI-Powered Criminal Network Analysis System

An interactive dashboard that turns police records (people + their interactions)
into a network and uses graph analytics and machine learning to:

1. **Identify key players**: leaders, brokers and inner-circle members
2. **Detect gangs / cells**: community detection
3. **Predict hidden links**: relationships missing from the records
4. **Score risk and flag anomalies**: who is most dangerous, and who behaves unusually
5. **Simulate arrests**: which arrests break the network apart most
6. **Evaluate the AI**: accuracy against known ground truth
7. **Monitor the network live**: new interactions stream in, scores update automatically, and suspicious events raise alerts

> The included dataset is **synthetic**. All names and records are randomly generated.

---

## How to run

```bash
pip install -r requirements.txt
python data/generate_sample_data.py      # optional, the app creates it automatically
streamlit run app.py
```

Or double-click `run.bat` on Windows. The dashboard opens at http://localhost:8501.

Check that everything works (prints accuracy of every module):

```bash
python tests/smoke_test.py
```

---

## Project structure

```
criminal-network-analysis/
├── app.py                      Streamlit dashboard (9 tabs)
├── requirements.txt
├── run.bat                     Windows launcher
├── data/
│   ├── generate_sample_data.py Synthetic dataset generator
│   ├── persons.csv             People (id, name, age, gender, city, prior_arrests)
│   ├── interactions.csv        Interactions (source, target, type, date, count)
│   └── ground_truth.csv        True gang and role (used only for evaluation)
├── src/
│   ├── graph_builder.py        Data validation and weighted graph construction
│   ├── analysis.py             Centrality metrics and key player ranking
│   ├── communities.py          Louvain / Greedy / Label Propagation community detection
│   ├── link_prediction.py      Random Forest hidden link prediction
│   ├── risk.py                 Risk score, Isolation Forest anomalies, role classifier
│   ├── disruption.py           Arrest (node removal) simulation
│   ├── live.py                 Live feeds (simulated / CSV watch) and alert engine
│   ├── visualize.py            Interactive network drawing (PyVis / vis.js)
│   └── pipeline.py             Runs all steps in order
├── tools/
│   └── push_events.py          Streams events into a CSV (stand-in for an external live source)
└── tests/
    └── smoke_test.py           End-to-end test with accuracy report
```

---

## System architecture

```
 CSV upload / sample data
          │
          ▼
 ┌──────────────────┐     validate, clean, filter by type/date
 │  Data layer       │
 └──────────────────┘
          │
          ▼
 ┌──────────────────┐     people = nodes, interactions = weighted edges
 │  Graph builder    │     weight = Σ count × type weight
 └──────────────────┘
          │
   ┌──────┼──────────────┬───────────────┬────────────────┐
   ▼      ▼              ▼               ▼                ▼
 SNA    Community     Link           Risk &           Disruption
 metrics detection    prediction     anomaly          simulation
 (6)    (Louvain)     (Random        (Isolation
                       Forest)        Forest)
   └──────┴──────────────┴───────────────┴────────────────┘
          │
          ▼
 ┌──────────────────┐
 │ Streamlit UI      │  graphs, tables, charts, CSV reports
 └──────────────────┘
```

---

## Modules and algorithms

### 1. Graph construction
Several interactions between the same two people are merged into one edge:

`weight = Σ (count × type_weight)`

| Interaction type        | Weight | Reason                     |
|-------------------------|--------|----------------------------|
| financial_transaction   | 3.0    | money is the strongest evidence |
| meeting                 | 2.5    | face-to-face contact       |
| co_arrest               | 2.0    | arrested together          |
| phone_call              | 1.0    |                            |
| social_media            | 0.5    | weakest evidence           |

`distance = 1 / weight` is used for shortest-path metrics, so a strong tie counts as a short distance.

### 2. Key player identification (Social Network Analysis)
| Metric       | What it finds |
|--------------|---------------|
| Degree       | people with the most contacts |
| Strength     | weighted degree |
| Betweenness  | **brokers** who sit on paths between groups |
| Closeness    | people who can reach everyone quickly |
| Eigenvector  | people connected to other important people |
| PageRank     | overall influence |

**Key player score (0–100)** = 30% PageRank + 25% betweenness + 20% eigenvector + 15% strength + 10% closeness (each min-max normalised).

### 3. Community detection
* **Louvain** (default): greedy modularity optimisation, fast and accurate
* **Greedy modularity** (Clauset-Newman-Moore)
* **Label propagation**

Quality is measured with **modularity** (above 0.3 means clear groups). Against ground truth it is measured with **NMI** and **ARI**.

### 4. Hidden link prediction (supervised ML)
1. Hide 20% of known links.
2. For each pair, compute features: common neighbours, Jaccard, Adamic-Adar, resource allocation, preferential attachment, same community, shortest path, same city, combined prior arrests.
3. Negatives = half random unlinked pairs, half "near misses" (2–3 hops apart), so the model learns the hard cases.
4. Train a **Random Forest** (300 trees) and evaluate on a 30% held-out split.
5. Score every unlinked pair within 3 hops and report the most probable hidden links.

### 5. Risk scoring and anomaly detection
* **Risk score** = 30% PageRank + 25% prior arrests + 20% betweenness + 15% financial links + 10% cross-community links, reported as Low / Medium / High / Critical.
* **Isolation Forest** (unsupervised) flags people whose feature profile is unusual, with a plain-language explanation.
* **Role classifier** (Random Forest, 5-fold cross-validation) learns to predict leader / lieutenant / broker / member / civilian from network features when labelled cases are available.

### 6. Live network monitoring (Live Monitor tab)
New interactions arrive every few seconds. Each event is added to the network, all key player
and risk scores are recalculated, and an **alert engine** checks the event:

| Alert | Trigger |
|-------|---------|
| Unknown person | someone not in the records contacts a known person (High if that person is high-risk, core or a broker) |
| Prediction confirmed | a hidden link the AI predicted actually appears, which validates the model live |
| Cross-group contact | first-ever contact between members of two different gangs |
| Large money transfer | 15 or more financial transactions in one event (Critical between two dangerous people) |
| Rising key player | someone enters the top 10 key players |

High and critical alerts pop up on screen. The event stream and the alert log can be downloaded as CSV.

**Data sources**
* **Simulated feed**: generates realistic events from the current network (for demos).
* **Watch CSV file**: reads new rows as they are appended to `data/live_interactions.csv`. This is how a
  real system (call-record export, bank alerts, FIR entries) would be connected. To try it:
  ```bash
  python tools/push_events.py        # appends events every second, like an external system would
  ```
  Then choose "Watch CSV file" in the Live Monitor tab and press Start.

Tick **"Include live events in all tabs"** in the sidebar to merge the live events into every other analysis.

### 7. Disruption simulation
Removes people one at a time using different strategies (key player score, betweenness, degree, risk score, random). After each removal it measures largest-group size, number of groups and **global efficiency** (how easily information can still flow). This shows which arrest order breaks the network apart fastest.

---

## Results on the sample dataset
(`python tests/smoke_test.py`; 137 people, 402 links, 6 gangs, 5 brokers)

| Module | Metric | Result |
|--------|--------|--------|
| Community detection (Louvain) | Modularity | 0.77 |
| Community detection | NMI / ARI vs true gangs | 1.00 / 1.00 |
| Hidden link prediction (Random Forest) | ROC-AUC | **0.89** |
| Best single heuristic (Adamic-Adar) | ROC-AUC | 0.72 |
| Role classifier | Accuracy / macro-F1 | 0.96 / 0.91 |

Community detection is perfect here because the synthetic gangs are built from strong
ties and the cross-gang noise is weak (calls and social media only). Real data would be
messier. Try the **"Ignore links weaker than"** slider and the different algorithms to
see how the results change.

---

## Using your own data

**persons.csv**: required `person_id, name`; optional `age, gender, city, prior_arrests`

**interactions.csv**: required `source, target, interaction_type`; optional `date` (YYYY-MM-DD), `count`

**ground_truth.csv** (optional, only for the Evaluation tab): `person_id, true_gang, true_role`

---

## Technology
Python 3.10+, Streamlit, NetworkX, scikit-learn, pandas, NumPy, PyVis (vis.js), Plotly

## Future scope
* Graph Neural Networks (GraphSAGE / GCN) for link prediction and role classification
* Connect the live monitor to a message queue (Kafka) or a police database instead of a CSV file
* NLP entity extraction from FIR / case report text to build the network automatically
* Geospatial crime mapping
* A graph database (Neo4j) for large-scale data
