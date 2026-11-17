import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Experiment Comparison", "Static vs periodic vs drift-aware retraining on identical streams, splits and model "
       "family — repeated across seeds and four drift types. Mean ± 95% CI.", badge="Controlled experiment", icon="🔬")

runs = data.runs()
summ = data.summary_json()
agg = pd.DataFrame(summ.get("aggregate", []))

# ---- hypothesis verdicts computed from data
piv = runs.pivot_table(index=["drift_type", "seed"], columns="policy", values="post_drift_auc").reset_index()
h1_wins = int((piv.groupby("drift_type").apply(lambda g: (g.drift_aware > g.static).mean() > 0.5, include_groups=False)).sum())
rt = runs.groupby(["drift_type", "policy"])["n_retrains"].mean().unstack()
h2_wins = int((rt["drift_aware"] < rt["periodic"]).sum())
n_dt = runs.drift_type.nunique()
fa = runs[runs.policy == "drift_aware"]["false_alarms"].mean()


def verdict(ok: float) -> tuple[str, str]:
    return ("Supported", t.SUCCESS) if ok >= 0.75 else ("Partially supported", t.WARNING) if ok >= 0.25 else ("Not supported", t.DANGER)


c1, c2, c3 = st.columns(3)
for col, (name, ratio, body) in zip(
    (c1, c2, c3),
    [("H1 · Faster recovery", h1_wins / n_dt, f"Drift-aware beats static post-drift AUC in <b>{h1_wins}/{n_dt}</b> drift types (majority of seeds)."),
     ("H2 · Fewer retrains", h2_wins / n_dt, f"Drift-aware needs fewer retrains than periodic in <b>{h2_wins}/{n_dt}</b> drift types."),
     ("H3 · Quality ≠ drift", 1.0 if fa < 0.5 else 0.5, f"Mean pre-drift false alarms: <b>{fa:.2f}</b> per stream despite injected quality faults.")]):
    v, c = verdict(ratio)
    with col:
        t.card(name, f"{t.chip(v, c)}<br><br>{body}", c)

t.section("Post-drift ROC-AUC", "error bars: 95% CI across seeds")
if not agg.empty:
    fig = go.Figure()
    for pol in ["static", "periodic", "drift_aware"]:
        a = agg[agg.policy == pol]
        fig.add_trace(go.Bar(x=a.drift_type, y=a.post_drift_auc_mean, name=pol, marker_color=t.POLICY_COLORS[pol],
                             error_y=dict(type="data", array=a.post_drift_auc_ci95.fillna(0), color="#fff", thickness=1.2)))
    fig.update_layout(barmode="group", yaxis_range=[0.5, 1.0])
    st.plotly_chart(t.style_fig(fig, 360), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    t.section("Per-seed distribution")
    fig = px.strip(runs, x="drift_type", y="post_drift_auc", color="policy", color_discrete_map=t.POLICY_COLORS)
    fig.update_traces(marker=dict(size=11, line=dict(color="#fff", width=1), opacity=0.9))
    st.plotly_chart(t.style_fig(fig, 340), use_container_width=True)
with c2:
    t.section("Cost vs. benefit", "retrains vs. post-drift AUC")
    fig = px.scatter(runs, x="n_retrains", y="post_drift_auc", color="policy", symbol="drift_type",
                     color_discrete_map=t.POLICY_COLORS)
    fig.update_traces(marker=dict(size=13, line=dict(color="#fff", width=1)))
    st.plotly_chart(t.style_fig(fig, 340), use_container_width=True)

t.section("Full results table", "mean ± 95% CI")
if not agg.empty:
    def fmt(m, c, d=3):
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return "—"
        return f"{m:.{d}f}" + (f" ± {c:.{d}f}" if c is not None and not (isinstance(c, float) and np.isnan(c)) else "")
    tbl = pd.DataFrame({
        "Drift": agg.drift_type, "Policy": agg.policy,
        "Post-drift AUC": [fmt(m, c) for m, c in zip(agg.post_drift_auc_mean, agg.post_drift_auc_ci95)],
        "Retrains": [fmt(m, c, 1) for m, c in zip(agg.n_retrains_mean, agg.n_retrains_ci95)],
        "Delay (records)": [fmt(m, c, 0) for m, c in zip(agg.detection_delay_mean, agg.detection_delay_ci95)],
        "Recovery (batches)": [fmt(m, c, 1) for m, c in zip(agg.recovery_batches_mean, agg.recovery_batches_ci95)],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download raw runs (CSV)", runs.to_csv(index=False), "sentinelml_runs.csv", "text/csv")

st.markdown('<div class="disclaimer"><b>Honest reading:</b> under <i>recurring</i> drift the old concept returns, so a static '
            'model can match or beat adaptive policies — periodic retraining chases the oscillation. Results are from a synthetic '
            'controlled stream; thresholds were fixed a-priori.</div>', unsafe_allow_html=True)
