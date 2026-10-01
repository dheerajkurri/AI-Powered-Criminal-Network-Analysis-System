"""
AI-Powered Criminal Network Analysis System
Run with:  streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src import analysis, communities, disruption, graph_builder, live, pipeline, risk
from src.analysis import METRIC_LABELS
from src.visualize import community_color, draw_network

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LIVE_FILE = os.path.join(DATA_DIR, "live_interactions.csv")
SEVERITY_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "⚪"}

st.set_page_config(page_title="Criminal Network Analysis", page_icon="🕸️", layout="wide")


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_sample():
    if not os.path.exists(os.path.join(DATA_DIR, "persons.csv")):
        from data.generate_sample_data import main as generate
        generate()
    return (
        pd.read_csv(os.path.join(DATA_DIR, "persons.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "interactions.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv")),
    )


@st.cache_data(show_spinner="Analysing network…")
def analyse(persons, interactions, types, start, end, min_weight, method):
    return pipeline.run(persons, interactions, types=list(types), start=start, end=end,
                        min_weight=min_weight, community_method=method)


@st.cache_data(show_spinner="Simulating arrests…")
def simulate(_G, scored, steps, key):
    return disruption.compare_strategies(_G, scored, steps=steps)


def person_label(G, pid):
    return f"{G.nodes[pid].get('name', pid)} ({pid})"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🕸️ Criminal Network Analysis")
source = st.sidebar.radio("Data source", ["Sample dataset", "Upload CSV files"])

truth = None
if source == "Sample dataset":
    persons, interactions, truth = load_sample()
else:
    st.sidebar.caption("persons.csv needs `person_id, name` (optional: age, gender, city, prior_arrests). "
                       "interactions.csv needs `source, target, interaction_type` (optional: date, count).")
    pf = st.sidebar.file_uploader("persons.csv", type="csv")
    inf = st.sidebar.file_uploader("interactions.csv", type="csv")
    tf = st.sidebar.file_uploader("ground_truth.csv (optional)", type="csv")
    if not (pf and inf):
        st.info("Upload both **persons.csv** and **interactions.csv** in the sidebar to begin, "
                "or switch to the sample dataset.")
        st.stop()
    persons, interactions = pd.read_csv(pf), pd.read_csv(inf)
    truth = pd.read_csv(tf) if tf else None

problems = graph_builder.validate(persons, interactions)
if problems:
    for p in problems:
        st.error(p)
    st.stop()

# Filters are built from the original records only, so live events never change them
prepared = graph_builder.prepare_interactions(interactions)
all_types = sorted(prepared["interaction_type"].unique())

use_live = st.sidebar.checkbox("Include live events in all tabs",
                               help="Merge events received in the Live Monitor tab into every other tab's analysis.")

st.sidebar.header("Filters")
types = st.sidebar.multiselect("Interaction types", all_types, default=all_types)
start = end = None
dmax = None
if prepared["date"].notna().any():
    dmin, dmax = prepared["date"].min().date(), prepared["date"].max().date()
    picked = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        start, end = picked
analysis_end = end

if use_live and st.session_state.get("live_events"):
    live_df = pd.DataFrame(st.session_state["live_events"])
    interactions = pd.concat([interactions, live_df], ignore_index=True)
    seen = set(live_df["source"].astype(str)) | set(live_df["target"].astype(str))
    unknown = sorted(seen - set(persons["person_id"].astype(str)))
    persons = pd.concat([persons, pd.DataFrame({"person_id": unknown, "name": [f"Unknown {u}" for u in unknown]})],
                        ignore_index=True)
    if end == dmax:
        analysis_end = None  # date range not narrowed, so keep live events (dated today)
    st.sidebar.caption(f"Including {len(live_df)} live events. Other tabs refresh when you interact with the page.")
min_weight = st.sidebar.slider("Ignore links weaker than", 0.0, 20.0, 0.0, 0.5,
                               help="Link weight = number of interactions × type weight "
                                    "(money 3, meeting 2.5, co-arrest 2, call 1, social media 0.5)")
method_label = st.sidebar.selectbox("Community detection algorithm", list(communities.ALGORITHMS))

if not types:
    st.warning("Select at least one interaction type.")
    st.stop()

R = analyse(persons, interactions, tuple(types), start, analysis_end, min_weight, communities.ALGORITHMS[method_label])
G, comm, scored = R["graph"], R["communities"], R["scored"]
if G.number_of_edges() == 0:
    st.warning("No links left after filtering. Relax the filters.")
    st.stop()

# ---------------------------------------------------------------- layout
st.title("AI-Powered Criminal Network Analysis System")
tabs = st.tabs([
    "📊 Overview", "📡 Live Monitor", "🕸️ Network", "🎯 Key Players", "👥 Communities",
    "🔮 Hidden Links", "⚠️ Risk & Anomalies", "🔎 Suspect Profile", "🚔 Disruption", "🧪 Model Evaluation",
])

# ---- Overview
with tabs[0]:
    s = R["summary"]
    cols = st.columns(4)
    for i, (k, v) in enumerate(s.items()):
        cols[i % 4].metric(k, v)
    c1, c2, c3 = st.columns(3)
    c1.metric("Communities found", len(set(comm.values())))
    c2.metric("Modularity", f"{R['modularity']:.3f}", help="> 0.3 = clear group structure")
    c3.metric("High / critical risk", int(scored["risk_level"].isin(["High", "Critical"]).sum()))

    left, right = st.columns(2)
    by_type = R["interactions"].groupby("interaction_type")["count"].sum().reset_index()
    left.plotly_chart(px.bar(by_type, x="interaction_type", y="count", title="Interactions by type"), use_container_width=True)
    if R["interactions"]["date"].notna().any():
        monthly = (R["interactions"].dropna(subset=["date"])
                   .groupby([pd.Grouper(key="date", freq="MS"), "interaction_type"])["count"].sum().reset_index())
        right.plotly_chart(px.line(monthly, x="date", y="count", color="interaction_type",
                                   title="Activity over time"), use_container_width=True)
    right_deg = pd.Series(dict(G.degree())).rename("contacts").reset_index()
    left.plotly_chart(px.histogram(right_deg, x="contacts", nbins=25, title="How many contacts each person has"),
                      use_container_width=True)
    right.plotly_chart(px.pie(scored, names="risk_level", title="Risk levels",
                              color="risk_level", color_discrete_map={"Critical": "#b00020", "High": "#f57c00",
                                                                      "Medium": "#fbc02d", "Low": "#43a047"}),
                       use_container_width=True)

# ---- Live monitor
def init_live(force: bool = False):
    """(Re)start live monitoring from the current analysed network."""
    ss = st.session_state
    key = (source, tuple(types), start, end, min_weight, method_label)
    if not force and ss.get("live_key") == key:
        return
    pl = R["predicted_links"]
    ss.update(
        live_key=key,
        live_G=G.copy(),
        live_base_nodes=set(G.nodes),
        live_comm=dict(comm),
        live_scored=scored,
        live_predicted={tuple(sorted((a, b))): p for a, b, p in zip(pl.get("person_a", []), pl.get("person_b", []),
                                                                     pl.get("probability", []))},
        live_events=[], live_alerts=[], live_history=[], live_new_edges=[],
        live_running=False,
        live_sim=live.SimulatedFeed(G, comm, pl, scored),
        live_file=live.FileFeed(LIVE_FILE),
    )


def live_view(feed_kind: str, rate: int):
    ss = st.session_state
    now = pd.Timestamp.now().strftime("%H:%M:%S")
    if ss.live_running:
        feed = ss.live_sim if feed_kind == "Simulated feed" else ss.live_file
        events = feed.next_events(rate)
        Gl, new_links, tick_alerts = ss.live_G, 0, []
        for e in events:
            tick_alerts += live.detect_alerts(e, Gl, ss.live_comm, ss.live_scored, ss.live_predicted, set(Gl.nodes))
            if live.add_event(Gl, e):
                new_links += 1
                ss.live_new_edges.append(tuple(sorted((str(e["source"]), str(e["target"])))))
            ss.live_events.append(e)
        if events:
            comm_live = {n: ss.live_comm.get(n, -1) for n in Gl}
            ranked = analysis.rank_key_players(analysis.compute_centrality(Gl), comm_live, Gl)
            rescored = risk.risk_scores(risk.node_feature_table(Gl, ranked))
            for msg in live.rank_changes(ss.live_scored, rescored):
                tick_alerts.append({"time": events[-1]["date"], "severity": "Medium", "type": "Rising key player",
                                    "details": msg, "source": "", "target": ""})
            ss.live_scored = rescored
            ss.live_alerts = tick_alerts[::-1] + ss.live_alerts
            for a in tick_alerts:
                if a["severity"] in ("Critical", "High"):
                    st.toast(f"{SEVERITY_ICON[a['severity']]} {a['type']}: {a['details']}")
        ss.live_history = (ss.live_history + [{"time": now, "events": len(events), "new_links": new_links,
                                               "alerts": len(tick_alerts)}])[-120:]

    Gl = ss.live_G
    alerts = pd.DataFrame(ss.live_alerts)
    serious = int(alerts["severity"].isin(["Critical", "High"]).sum()) if not alerts.empty else 0
    st.markdown(f"{'🟢 **LIVE**' if ss.live_running else '⏸️ **Paused**'} · last update {now} · "
                f"network now has {Gl.number_of_nodes()} people and {Gl.number_of_edges()} links")
    m = st.columns(5)
    m[0].metric("Events received", len(ss.live_events))
    m[1].metric("New links", len(ss.live_new_edges))
    m[2].metric("Alerts", len(alerts))
    m[3].metric("High / critical alerts", serious)
    m[4].metric("Unknown people", len(set(Gl.nodes) - ss.live_base_nodes))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Recent activity** (people in the last 40 events · orange = new link)")
        recent = ss.live_events[-40:]
        if recent:
            active = {str(e["source"]) for e in recent} | {str(e["target"]) for e in recent}
            comm_live = {n: ss.live_comm.get(n) for n in Gl}
            components.html(draw_network(Gl.subgraph(active), comm_live, ss.live_scored, height="460px",
                                         new_edges=set(ss.live_new_edges[-40:])), height=490)
        else:
            st.info("Press **▶ Start** to begin receiving events.")
    with right:
        st.markdown("**Alerts** (newest first)")
        if alerts.empty:
            st.write("No alerts yet.")
        else:
            view = alerts.head(50).copy()
            view.insert(0, "", view["severity"].map(SEVERITY_ICON))
            view["time"] = view["time"].astype(str).str[-8:]
            st.dataframe(view[["", "time", "type", "details"]], hide_index=True, use_container_width=True, height=460)

    if ss.live_history:
        hist = pd.DataFrame(ss.live_history)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(hist, x="time", y=["events", "new_links", "alerts"], barmode="group",
                              title="Activity per update"), use_container_width=True)
        base_rank = scored.set_index("person_id")["rank"]
        top = ss.live_scored.nsmallest(10, "rank")[["rank", "person_id", "name", "key_player_score", "risk_level"]].copy()
        top["change"] = top.apply(lambda r: "🆕" if r.person_id not in base_rank.index else
                                  (f"▲ {int(base_rank[r.person_id] - r['rank'])}" if base_rank[r.person_id] > r["rank"] else
                                   f"▼ {int(r['rank'] - base_rank[r.person_id])}" if base_rank[r.person_id] < r["rank"] else "–"),
                                  axis=1)
        b.markdown("**Live top 10 key players** (change vs. start)")
        b.dataframe(top, hide_index=True, use_container_width=True)

    if ss.live_events:
        st.markdown("**Latest events**")
        st.dataframe(pd.DataFrame(ss.live_events[-15:][::-1]), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download live events (CSV)", pd.DataFrame(ss.live_events).to_csv(index=False),
                           "live_events.csv", "text/csv")
        if not alerts.empty:
            c2.download_button("⬇️ Download alert log (CSV)", alerts.to_csv(index=False), "alerts.csv", "text/csv")


with tabs[1]:
    init_live()
    ss = st.session_state
    st.subheader("Real-time network monitoring")
    st.caption("New interactions stream in, the network and all scores update automatically, and the alert engine "
               "flags unknown people, first contacts between gangs, large money transfers, confirmed AI "
               "predictions and people rising into the top key players.")
    c = st.columns([1.4, 1.2, 1.2, 0.8, 0.8])
    feed_kind = c[0].radio("Data source", ["Simulated feed", "Watch CSV file"], horizontal=True)
    rate = c[1].slider("Events per update", 1, 10, 3)
    interval = c[2].select_slider("Update every (seconds)", [1, 2, 3, 5, 10], value=2)
    c[3].write("")
    if c[3].button("⏸ Pause" if ss.live_running else "▶ Start", use_container_width=True, type="primary"):
        ss.live_running = not ss.live_running
        st.rerun()
    c[4].write("")
    if c[4].button("↺ Reset", use_container_width=True):
        init_live(force=True)
        st.rerun()
    if feed_kind == "Watch CSV file":
        st.info(f"Watching `{LIVE_FILE}` for new rows. To stream records into it, run this in a terminal:  \n"
                "`python tools/push_events.py`")
    st.fragment(run_every=interval if ss.live_running else None)(live_view)(feed_kind, rate)

# ---- Network
with tabs[2]:
    st.caption("Colour = community · size = key player score · ★ core · ◆ broker · ▲ inner circle · "
               "red dashed = AI-predicted hidden link. Hover for details, scroll to zoom, drag to move.")
    c1, c2, c3 = st.columns([2, 1, 1])
    focus = c1.selectbox("Focus on a person (optional)", ["Whole network"] + [person_label(G, n) for n in scored["person_id"]])
    radius = c2.slider("Hops around person", 1, 3, 2)
    show_pred = c3.checkbox("Show predicted links", True)
    n_pred = c3.number_input("How many", 1, 50, 10) if show_pred else 0

    H, highlight = G, set()
    if focus != "Whole network":
        pid = focus.rsplit("(", 1)[1].rstrip(")")
        H, highlight = analysis.ego_network(G, pid, radius), {pid}
    pred = R["predicted_links"].head(int(n_pred)) if show_pred and not R["predicted_links"].empty else None
    components.html(draw_network(H, comm, scored, highlight, pred), height=680, scrolling=False)

# ---- Key players
with tabs[3]:
    st.subheader("Who matters most in the network?")
    st.caption("Key player score (0-100) combines PageRank 30%, betweenness 25%, eigenvector 20%, "
               "strength 15% and closeness 10%. Position is inferred from network shape.")
    top_n = st.slider("Show top", 5, len(scored), min(20, len(scored)))
    view = scored.sort_values("rank").head(top_n)
    st.dataframe(view[["rank", "person_id", "name", "key_player_score", "predicted_role", "community",
                       "degree", "betweenness", "pagerank", "communities_linked"]],
                 hide_index=True, use_container_width=True,
                 column_config={"key_player_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f")})
    metric = st.selectbox("Compare by metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
    top_m = scored.nlargest(15, metric)
    st.plotly_chart(px.bar(top_m, x=metric, y="name", orientation="h", color="predicted_role",
                           title=f"Top 15 by {METRIC_LABELS[metric]}").update_yaxes(autorange="reversed"),
                    use_container_width=True)
    st.download_button("⬇️ Download key player report (CSV)", scored.to_csv(index=False), "key_players.csv", "text/csv")

# ---- Communities
with tabs[4]:
    st.subheader(f"Detected groups — {method_label}")
    cs = R["community_summary"]
    st.dataframe(cs, hide_index=True, use_container_width=True)
    st.plotly_chart(px.bar(cs, x="community", y="size", color=cs["community"].map(community_color),
                           color_discrete_map="identity", title="Community sizes"), use_container_width=True)
    cid = st.selectbox("Inspect community", cs["community"].tolist())
    members = [n for n, c in comm.items() if c == cid]
    st.dataframe(scored[scored["person_id"].isin(members)][["person_id", "name", "key_player_score", "predicted_role",
                                                            "risk_level", "prior_arrests"]],
                 hide_index=True, use_container_width=True)
    components.html(draw_network(G.subgraph(members), comm, scored), height=520)

# ---- Hidden links
with tabs[5]:
    st.subheader("AI-predicted hidden relationships")
    lm = R["link_model"]
    if lm is None:
        st.info("Network is too small to train the link prediction model (needs at least 30 links).")
    else:
        st.caption("A Random Forest learns what a real link looks like by hiding 20% of known links and "
                   "learning to find them again. It then scores every unconnected pair within 3 hops.")
        cols = st.columns(4)
        for i, k in enumerate(["ROC-AUC", "Average precision", "Precision", "Recall"]):
            cols[i].metric(k, lm.metrics[k])
        st.dataframe(R["predicted_links"], hide_index=True, use_container_width=True,
                     column_config={"probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="%.2f")})
        left, right = st.columns(2)
        base = pd.DataFrame({"method": list(lm.baseline_auc) + ["Random Forest (ours)"],
                             "ROC-AUC": list(lm.baseline_auc.values()) + [lm.metrics["ROC-AUC"]]})
        left.plotly_chart(px.bar(base, x="method", y="ROC-AUC", range_y=[0.4, 1], title="Our model vs classic heuristics"),
                          use_container_width=True)
        right.plotly_chart(px.bar(lm.feature_importance, x="importance", y="feature", orientation="h",
                                  title="What the model looks at").update_yaxes(autorange="reversed"),
                           use_container_width=True)
        st.download_button("⬇️ Download predicted links (CSV)", R["predicted_links"].to_csv(index=False),
                           "predicted_links.csv", "text/csv")

# ---- Risk
with tabs[6]:
    st.subheader("Risk assessment")
    st.caption("Risk = influence 30% + criminal history 25% + brokerage 20% + money links 15% + "
               "cross-group links 10%. Anomalies are flagged by an Isolation Forest.")
    level = st.multiselect("Risk level", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
    st.dataframe(scored[scored["risk_level"].isin(level)][["person_id", "name", "risk_score", "risk_level",
                                                           "prior_arrests", "financial_links", "predicted_role", "anomaly"]],
                 hide_index=True, use_container_width=True,
                 column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f")})
    st.plotly_chart(px.scatter(scored, x="key_player_score", y="prior_arrests", size="financial_links",
                               color="risk_level", hover_name="name", symbol="anomaly",
                               title="Network importance vs criminal history (size = money links)"),
                    use_container_width=True)
    st.markdown("#### Unusual behaviour (anomalies)")
    anomalies = scored[scored["anomaly"]].copy()
    if anomalies.empty:
        st.write("No anomalies detected.")
    else:
        anomalies["why"] = anomalies.apply(lambda r: risk.explain_anomaly(r, scored), axis=1)
        st.dataframe(anomalies[["person_id", "name", "anomaly_score", "risk_level", "why"]], hide_index=True, use_container_width=True)

# ---- Suspect profile
with tabs[7]:
    pick = st.selectbox("Select suspect", [person_label(G, n) for n in scored["person_id"]])
    pid = pick.rsplit("(", 1)[1].rstrip(")")
    row = scored.set_index("person_id").loc[pid]
    attrs = G.nodes[pid]
    c = st.columns(5)
    c[0].metric("Rank", int(row["rank"]))
    c[1].metric("Key player score", row["key_player_score"])
    c[2].metric("Risk", f"{row['risk_score']} ({row['risk_level']})")
    c[3].metric("Position", row["predicted_role"])
    c[4].metric("Community", int(row["community"]))
    st.write(f"**City:** {attrs.get('city', '-')} · **Age:** {attrs.get('age', '-')} · "
             f"**Prior arrests:** {attrs.get('prior_arrests', '-')} · **Anomalous:** {'Yes' if row['anomaly'] else 'No'}")

    left, right = st.columns([3, 2])
    with left:
        components.html(draw_network(analysis.ego_network(G, pid, 2), comm, scored, {pid}), height=480)
    with right:
        contacts = pd.DataFrame([{
            "contact": person_label(G, m),
            "weight": round(G[pid][m]["weight"], 1),
            "interactions": G[pid][m]["interactions"],
            "types": ", ".join(G[pid][m]["types"]),
        } for m in G.neighbors(pid)]).sort_values("weight", ascending=False)
        st.markdown("**Direct contacts**")
        st.dataframe(contacts, hide_index=True, use_container_width=True, height=250)
        pl = R["predicted_links"]
        if not pl.empty:
            mine = pl[(pl["person_a"] == pid) | (pl["person_b"] == pid)]
            st.markdown("**Predicted hidden contacts**")
            st.dataframe(mine[["name_a", "name_b", "probability"]] if not mine.empty else pd.DataFrame({"": ["None in top list"]}),
                         hide_index=True, use_container_width=True)

    st.markdown("#### Trace the connection between two people")
    other = st.selectbox("Second person", [person_label(G, n) for n in scored["person_id"] if n != pid])
    path = analysis.shortest_connection(G, pid, other.rsplit("(", 1)[1].rstrip(")"))
    st.write(" → ".join(person_label(G, n) for n in path) if path else "No connection found.")

# ---- Disruption
with tabs[8]:
    st.subheader("If we arrest these people, how much does the network break apart?")
    steps = st.slider("Number of arrests", 1, min(30, G.number_of_nodes() - 1), min(15, G.number_of_nodes() - 1))
    sim = simulate(G, scored, steps, (tuple(types), start, end, min_weight, method_label))
    left, right = st.columns(2)
    left.plotly_chart(px.line(sim, x="removed", y="largest_group_pct", color="strategy", markers=True,
                              title="Size of largest connected group (%)", labels={"removed": "People arrested"}),
                      use_container_width=True)
    right.plotly_chart(px.line(sim, x="removed", y="efficiency", color="strategy", markers=True,
                               title="Communication efficiency", labels={"removed": "People arrested"}),
                       use_container_width=True)
    final = sim[sim["removed"] == steps].sort_values("efficiency")
    best = final.iloc[0]
    st.success(f"Best strategy for {steps} arrests: **{best['strategy']}** — communication efficiency drops to "
               f"{best['efficiency']:.3f} and the network splits into {int(best['groups'])} groups.")
    st.dataframe(final[["strategy", "largest_group_pct", "groups", "efficiency"]], hide_index=True, use_container_width=True)
    order = sim[(sim["strategy"] == best["strategy"]) & sim["person_id"].notna()]["person_id"]
    st.markdown("**Recommended arrest order:** " + " → ".join(G.nodes[p].get("name", p) for p in order))

# ---- Evaluation
with tabs[9]:
    st.subheader("How accurate is the AI? (needs ground truth labels)")
    if truth is None:
        st.info("Upload a ground_truth.csv with `person_id, true_gang, true_role` to evaluate the models.")
    else:
        ev = communities.evaluate_against_truth(comm, truth)
        c = st.columns(3)
        c[0].metric("Community NMI", ev.get("NMI", "-"), help="1.0 = detected groups exactly match real gangs")
        c[1].metric("Community ARI", ev.get("ARI", "-"))
        if R["link_model"]:
            c[2].metric("Hidden link ROC-AUC", R["link_model"].metrics["ROC-AUC"])

        st.markdown("#### Supervised role classifier (Random Forest, 5-fold cross-validation)")
        rc = risk.train_role_classifier(scored, truth)
        if rc is None:
            st.info("Not enough labelled examples of each role to train the classifier.")
        else:
            c = st.columns(2)
            c[0].metric("Accuracy", rc["accuracy"])
            c[1].metric("Macro F1", rc["macro_f1"])
            left, right = st.columns(2)
            left.plotly_chart(px.imshow(rc["confusion"], text_auto=True, color_continuous_scale="Blues",
                                        title="Confusion matrix"), use_container_width=True)
            right.plotly_chart(px.bar(rc["importance"], x="importance", y="feature", orientation="h",
                                      title="Most useful features").update_yaxes(autorange="reversed"),
                               use_container_width=True)
            st.dataframe(rc["report"], use_container_width=True)
