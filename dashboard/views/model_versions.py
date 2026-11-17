import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

t.hero("Model Versions", "Champion / candidate registry. A candidate is promoted only if it beats the champion on "
       "held-out recent data by the configured margin and meets the latency budget.", badge="Registry", icon="🗂️")

mv = data.model_versions()
if mv.empty:
    st.info("Registry is empty.")
    st.stop()

mv["roc_auc"] = mv.metrics.apply(lambda m: m.get("roc_auc"))
mv["p95_ms"] = mv.metrics.apply(lambda m: m.get("p95_ms"))
status_color = {"champion": t.SUCCESS, "archived": t.MUTED, "rejected": t.DANGER, "candidate": t.WARNING}

t.kpis([
    ("Versions", len(mv), None, "🗂️", t.PRIMARY),
    ("Promoted", int((mv.status.isin(["champion", "archived"])).sum()), None, "✅", t.SUCCESS),
    ("Rejected", int((mv.status == "rejected").sum()), "failed validation gate", "⛔", t.DANGER),
    ("Current champion", mv[mv.status == "champion"].version.iloc[0] if (mv.status == "champion").any() else "—", None, "🏆", t.ACCENT),
])

t.section("Version timeline")
fig = go.Figure()
for stt, g in mv.groupby("status"):
    fig.add_trace(go.Scatter(x=g.version, y=g.roc_auc, mode="markers+text", name=stt, text=g.version,
                             textposition="top center",
                             marker=dict(size=22, color=status_color.get(stt, t.PRIMARY), line=dict(color="#fff", width=1.5))))
fig.add_trace(go.Scatter(x=mv.version, y=mv.roc_auc, mode="lines", line=dict(color=t.BORDER, dash="dot"), showlegend=False))
fig.update_layout(yaxis_title="validation ROC-AUC", xaxis_title="version")
st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)

t.section("Registry entries")
for r in mv.iloc[::-1].itertuples():
    st.markdown(
        f"""<div class="tl-item" style="--sev:{status_color.get(r.status, t.PRIMARY)}">
        <div class="meta">{t.chip(r.status, status_color.get(r.status, t.PRIMARY))} &nbsp; <b>{r.version}</b> · {r.algorithm} ·
        trained on {int(r.trained_on_records):,} records · {r.created_at}</div>
        <div class="msg">AUC {r.roc_auc:.3f} · p95 {r.p95_ms or 0:.2f} ms — {r.reason}</div></div>""",
        unsafe_allow_html=True,
    )

with st.expander("Raw registry table"):
    st.dataframe(mv.drop(columns=["metrics"]), use_container_width=True, hide_index=True)
