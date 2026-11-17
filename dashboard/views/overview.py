import numpy as np
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("SentinelML", "Observes incoming data, catches quality failures and distribution shift, measures model "
       "degradation and tests whether drift-aware retraining restores reliability — with every number traced to a logged batch.",
       badge="Monitoring active")

runs = data.runs()
rid = data.showcase_run_id()
bm = data.batch_metrics(rid) if rid else None
mv = data.model_versions()
ev = data.events(rid) if rid else None

champ = mv[mv.status == "champion"]["version"].iloc[0] if not mv.empty and (mv.status == "champion").any() else "—"
sud = runs[runs.drift_type == "sudden"].groupby("policy")["post_drift_auc"].mean()
da, st_ = sud.get("drift_aware", np.nan), sud.get("static", np.nan)
last_q = bm["quality_score"].iloc[-1] if bm is not None and len(bm) else np.nan

t.kpis([
    ("Champion model", champ, f"{len(mv)} versions in registry", "🏆", t.PRIMARY),
    ("Post-drift AUC", f"{da:.3f}", f"{da - st_:+.3f} vs static", "📈", t.SUCCESS),
    ("Retrains (sudden)", f"{int(runs[(runs.drift_type=='sudden')&(runs.policy=='drift_aware')]['n_retrains'].mean())}",
     f"periodic: {int(runs[(runs.drift_type=='sudden')&(runs.policy=='periodic')]['n_retrains'].mean())}", "🔁", t.ACCENT),
    ("Latest quality score", f"{last_q:.0f}/100", None, "🧪", t.SECONDARY),
    ("Events logged", f"{len(ev) if ev is not None else 0}", "drift · quality · promotion", "⚡", t.WARNING),
])

t.section("System pipeline", "every stage is a tested module in src/sentinelml")
t.flow([("Ingest", "CSV · API · replay"), ("Validate", "schema · ranges · nulls"), ("Predict", "champion model"),
        ("Monitor", "PSI · KS · Page-Hinkley"), ("Retrain", "candidate vs champion"), ("Explain", "permutation · local")])

c1, c2 = st.columns([1.6, 1])
with c1:
    t.section("Showcase run", "sudden drift · drift-aware policy · seed 0")
    if bm is not None:
        fig = go.Figure()
        db_ = data.drift_batch()
        fig.add_vrect(x0=db_ - 0.5, x1=bm.batch_id.max() + 0.5, fillcolor=t.DANGER, opacity=0.06, line_width=0,
                      annotation_text="post-drift", annotation_position="top left")
        fig.add_trace(go.Scatter(x=bm.batch_id, y=bm.roc_auc, name="ROC-AUC", mode="lines+markers",
                                 line=dict(color=t.PRIMARY, width=3, shape="spline"), fill="tozeroy",
                                 fillcolor="rgba(99,102,241,0.10)"))
        fig.add_trace(go.Scatter(x=bm.batch_id, y=bm.quality_score / 100, name="Quality (÷100)",
                                 line=dict(color=t.SECONDARY, dash="dot")))
        rt = bm[bm.retrained == 1]
        fig.add_trace(go.Scatter(x=rt.batch_id, y=rt.roc_auc, mode="markers", name="Promotion",
                                 marker=dict(symbol="star", size=16, color=t.WARNING, line=dict(color="#fff", width=1))))
        fig.update_yaxes(range=[0.4, 1.02])
        st.plotly_chart(t.style_fig(fig, 360), use_container_width=True)
with c2:
    t.section("Recent events")
    if ev is not None:
        t.timeline(ev.to_dict("records"), limit=6)

t.section("Policy snapshot", "mean post-drift ROC-AUC across seeds")
agg = runs.groupby(["drift_type", "policy"])["post_drift_auc"].mean().reset_index()
fig = go.Figure()
for pol in ["static", "periodic", "drift_aware"]:
    s = agg[agg.policy == pol]
    fig.add_trace(go.Bar(x=s.drift_type, y=s.post_drift_auc, name=pol, marker_color=t.POLICY_COLORS[pol],
                         text=s.post_drift_auc.round(3), textposition="outside"))
fig.update_layout(barmode="group", yaxis_range=[0.5, 1.0])
st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    t.card("H1 · Faster recovery", "Drift-aware retraining should recover predictive performance faster than a static model after a controlled drift event.")
with c2:
    t.card("H2 · Fewer retrains", "Retraining only on validated drift should need fewer training runs than a fixed schedule.", t.ACCENT)
with c3:
    t.card("H3 · Quality ≠ drift", "Quality checks should catch preventable input faults before inference — and not be confused with real drift.", t.SECONDARY)
