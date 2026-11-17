"""Animated replay of a logged run, batch by batch."""
import time

import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Live Stream", "Replay any logged run as if it were arriving live: predictions, quality, drift statistics "
       "and policy decisions appear batch by batch.", badge="Replay mode", icon="📡")

runs = data.runs()
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
dt = c1.selectbox("Drift type", sorted(runs.drift_type.unique()), index=sorted(runs.drift_type.unique()).index("sudden"))
pol = c2.selectbox("Policy", ["drift_aware", "periodic", "static"])
seed = c3.selectbox("Seed", sorted(runs.seed.unique()))
speed = c4.slider("Speed (batches/s)", 1, 20, 8)

rid = data.run_id_for(dt, pol, int(seed))
bm = data.batch_metrics(rid)
ev = data.events(rid)

kpi_slot = st.empty()
chart_slot = st.empty()
lc, rc = st.columns([1.5, 1])
psi_slot = lc.empty()
log_slot = rc.empty()


def render(upto: int) -> None:
    cur = bm.iloc[: upto + 1]
    last = cur.iloc[-1]
    with kpi_slot.container():
        t.kpis([
            ("Batch", f"{int(last.batch_id)}", f"{len(cur)}/{len(bm)} replayed", "🧭", t.PRIMARY),
            ("ROC-AUC", f"{last.roc_auc:.3f}", f"rolling {last.rolling_auc:.3f}", "📈", t.SUCCESS),
            ("Max PSI", f"{last.max_psi:.2f}", "▲ drift" if last.drift_flag else "stable", "🌊", t.WARNING),
            ("Quality", f"{last.quality_score:.0f}", None, "🧪", t.SECONDARY),
            ("Model", str(last.model_version), f"{int(cur.retrained.sum())} promotions", "🏷️", t.ACCENT),
        ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cur.batch_id, y=cur.roc_auc, name="ROC-AUC", line=dict(color=t.PRIMARY, width=3, shape="spline")))
    fig.add_trace(go.Scatter(x=cur.batch_id, y=cur.rolling_auc, name="Rolling AUC", line=dict(color=t.SECONDARY, dash="dash")))
    if "concept_alpha" in cur and cur.concept_alpha.notna().any():
        fig.add_trace(go.Scatter(x=cur.batch_id, y=cur.concept_alpha, name="Ground-truth drift α", yaxis="y2",
                                 line=dict(color=t.DANGER, width=1), fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
    flags = cur[cur.drift_flag == 1]
    fig.add_trace(go.Scatter(x=flags.batch_id, y=flags.roc_auc, mode="markers", name="Drift alarm",
                             marker=dict(color=t.WARNING, size=10, symbol="diamond")))
    rt = cur[cur.retrained == 1]
    fig.add_trace(go.Scatter(x=rt.batch_id, y=rt.roc_auc, mode="markers", name="Model update",
                             marker=dict(color="#fff", size=14, symbol="star")))
    fig.update_layout(xaxis=dict(range=[bm.batch_id.min() - 0.5, bm.batch_id.max() + 0.5], title="batch"),
                      yaxis=dict(range=[0.4, 1.02], title="ROC-AUC"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], showgrid=False, title="α"))
    chart_slot.plotly_chart(t.style_fig(fig, 380), use_container_width=True, key=f"live_{upto}")
    pf = go.Figure(go.Bar(x=cur.batch_id.tail(20), y=cur.max_psi.tail(20),
                          marker=dict(color=cur.max_psi.tail(20), colorscale=[[0, t.SUCCESS], [0.3, t.WARNING], [1, t.DANGER]],
                                      cmin=0, cmax=1)))
    pf.add_hline(y=data.settings.monitoring["drift"]["psi_threshold"], line_dash="dot", line_color=t.DANGER,
                 annotation_text="PSI threshold")
    psi_slot.plotly_chart(t.style_fig(pf, 260, "Max feature PSI (last 20 batches)"), use_container_width=True, key=f"psi_{upto}")
    with log_slot.container():
        recent = ev[ev.batch_id <= last.batch_id]
        t.timeline(recent.to_dict("records"), limit=5)


b1, b2 = st.columns([1, 5])
play = b1.button("▶ Start replay", use_container_width=True)
pos = b2.slider("Scrub to batch", 0, len(bm) - 1, len(bm) - 1, label_visibility="collapsed")

if play:
    for i in range(len(bm)):
        render(i)
        time.sleep(1 / speed)
else:
    render(pos)
