import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data
from sentinelml.common.config import PROJECT_ROOT
from sentinelml.validation.quality import run_quality_checks

t.hero("Data Quality", "Schema, range, null, duplicate, category and outlier checks run on every batch before inference. "
       "Faulted batches are cleaned and gated so they are not mistaken for genuine drift.", badge="Quality gate", icon="🧪")

rid = data.showcase_run_id()
bm = data.batch_metrics(rid)
qev = data.events(rid, "quality")

c1, c2, c3, c4 = st.columns([1, 1, 1, 2.2])
with c1:
    t.ring(bm.quality_score.mean(), "Mean quality score")
with c2:
    t.ring(100 * (bm.quality_score >= 80).mean(), "Batches passing", t.PRIMARY)
with c3:
    t.ring(bm.quality_score.min(), "Worst batch")
with c4:
    fig = go.Figure(go.Scatter(x=bm.batch_id, y=bm.quality_score, mode="lines+markers", fill="tozeroy",
                               line=dict(color=t.SECONDARY, width=2, shape="spline"), fillcolor="rgba(34,211,238,.08)"))
    fig.add_hline(y=80, line_dash="dot", line_color=t.WARNING, annotation_text="pass threshold")
    bad = bm[bm.quality_score < 80]
    fig.add_trace(go.Scatter(x=bad.batch_id, y=bad.quality_score, mode="markers",
                             marker=dict(size=14, color=t.DANGER, symbol="x"), name="gated"))
    fig.update_layout(showlegend=False, yaxis_range=[0, 105])
    st.plotly_chart(t.style_fig(fig, 220, "Quality score per batch"), use_container_width=True)

t.section("Quality incidents", "from the event log")
if qev.empty:
    st.info("No quality incidents recorded.")
for r in qev.itertuples():
    with st.expander(f"Batch {r.batch_id} · {r.message}"):
        for msg in r.details.get("issues", []):
            st.markdown(f"- {msg}")

t.section("Interactive validator", "upload a CSV or use the shipped sample")
src = st.radio("Source", ["Clean sample", "Faulted sample", "Upload CSV"], horizontal=True)
df = None
if src == "Clean sample":
    df = pd.read_csv(PROJECT_ROOT / "data/sample/stream_sample.csv")
elif src == "Faulted sample":
    df = pd.read_csv(PROJECT_ROOT / "data/sample/quality_fault_sample.csv")
else:
    up = st.file_uploader("CSV with the configured schema", type="csv")
    if up:
        df = pd.read_csv(up)

if df is not None:
    rep = run_quality_checks(df, data.settings.data["schema"])
    t.kpis([
        ("Rows", f"{rep.n_rows:,}", None, "📄", t.PRIMARY),
        ("Score", f"{rep.score:.0f}/100", "▲ pass" if rep.passed else "▼ fail", "🧪", t.SUCCESS if rep.passed else t.DANGER),
        ("Issues", f"{len(rep.issues)}", None, "⚠️", t.WARNING),
    ])
    if rep.issues:
        idf = pd.DataFrame([i.__dict__ for i in rep.issues])
        idf["rate"] = (idf["rate"] * 100).round(2).astype(str) + "%"
        st.dataframe(idf, use_container_width=True, hide_index=True)
    with st.expander("Preview data"):
        st.dataframe(df.head(50), use_container_width=True)
