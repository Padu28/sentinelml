import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data
from sentinelml.ingestion.loaders import load_stream
from sentinelml.monitoring.drift import psi_categorical, psi_numeric
from sentinelml.validation.quality import clean_batch

t.hero("Drift Events", "Reference-vs-current window tests (PSI + Bonferroni-corrected KS) and a Page-Hinkley "
       "detector on the error rate. Detection delay and false alarms are measured against known ground truth.",
       badge="Drift lab", icon="🌊")

runs = data.runs()
s = data.settings
bs = s.data["stream"]["batch_size"]

dt = st.segmented_control("Drift type", sorted(runs.drift_type.unique()), default="sudden")
dt = dt or "sudden"
sub = runs[(runs.drift_type == dt) & (runs.policy == "drift_aware")]
t.kpis([
    ("Detection delay", f"{sub.detection_delay.mean():.0f} rec", f"± {sub.detection_delay.std():.0f} across seeds", "⏱️", t.PRIMARY),
    ("False alarms", f"{sub.false_alarms.mean():.1f}", "pre-drift alarms per stream", "🚨", t.WARNING),
    ("Drift onset", f"batch {data.drift_batch()}", f"record {s.data['synthetic']['drift_start']:,}", "🎯", t.DANGER),
    ("Seeds", f"{sub.seed.nunique()}", None, "🎲", t.ACCENT),
])


@st.cache_data
def psi_matrix(drift_type: str, seed: int) -> pd.DataFrame:
    df = load_stream(drift_type, seed)
    schema = s.data["schema"]
    n_train = s.data["stream"]["train_records"] // bs
    ref = clean_batch(df[df.batch_id < n_train], schema)
    rows = []
    for b, g in df[df.batch_id >= n_train].groupby("batch_id"):
        g = clean_batch(g, schema)
        r = {"batch_id": b}
        for f in s.numeric_features:
            r[f] = psi_numeric(ref[f].to_numpy(float), g[f].to_numpy(float))
        for f in s.categorical_features:
            r[f] = psi_categorical(ref[f].astype(str), g[f].astype(str))
        rows.append(r)
    return pd.DataFrame(rows).set_index("batch_id")


t.section("Feature PSI heatmap", "vs. the initial training reference — no retraining applied")
m = psi_matrix(dt, int(runs.seed.min()))
fig = go.Figure(go.Heatmap(z=m.T.values, x=m.index, y=m.columns, zmin=0, zmax=1.0,
                           colorscale=[[0, "#0f172a"], [0.2, t.PRIMARY], [0.5, t.ACCENT], [1, t.DANGER]],
                           colorbar=dict(title="PSI"), hovertemplate="batch %{x}<br>%{y}: PSI %{z:.3f}<extra></extra>"))
fig.add_vline(x=data.drift_batch() - 0.5, line_color="#fff", line_dash="dash", annotation_text="drift onset")
st.plotly_chart(t.style_fig(fig, 330), use_container_width=True)

c1, c2 = st.columns([1.2, 1])
with c1:
    t.section("Alarm timeline", "all seeds · drift-aware runs")
    allb = data.all_batch_metrics()
    a = allb[(allb.drift_type == dt) & (allb.policy == "drift_aware")]
    fig = go.Figure()
    for seed_, g in a.groupby("seed"):
        fl = g[g.drift_flag == 1]
        fig.add_trace(go.Scatter(x=fl.batch_id, y=[f"seed {seed_}"] * len(fl), mode="markers", name=f"seed {seed_}",
                                 marker=dict(size=12, symbol="diamond", color=t.WARNING, line=dict(color="#fff", width=1))))
        rt = g[g.retrained == 1]
        fig.add_trace(go.Scatter(x=rt.batch_id, y=[f"seed {seed_}"] * len(rt), mode="markers", showlegend=False,
                                 marker=dict(size=16, symbol="star", color=t.SUCCESS)))
    fig.add_vline(x=data.drift_batch() - 0.5, line_color=t.DANGER, line_dash="dash")
    fig.update_layout(showlegend=False, xaxis_title="batch  (◆ alarm · ★ promotion)")
    st.plotly_chart(t.style_fig(fig, 280), use_container_width=True)
with c2:
    t.section("Latest drift events")
    ev = data.events(data.run_id_for(dt, "drift_aware", int(runs.seed.min())), "drift")
    t.timeline(ev.to_dict("records"), limit=5)

t.section("Feature-level detail", "pick an alarm to inspect KS statistics")
ev_all = data.events(data.run_id_for(dt, "drift_aware", int(runs.seed.min())), "drift")
if not ev_all.empty:
    choice = st.selectbox("Alarm", ev_all.batch_id.tolist(), format_func=lambda b: f"batch {b}")
    det = ev_all[ev_all.batch_id == choice].iloc[0].details
    fdf = pd.DataFrame(det.get("features", []))
    if not fdf.empty:
        fdf["p_value"] = fdf["p_value"].apply(lambda v: f"{v:.2e}" if v is not None and not np.isnan(v) else "—")
        st.dataframe(fdf, use_container_width=True, hide_index=True,
                     column_config={"psi": st.column_config.ProgressColumn("PSI", min_value=0, max_value=1.5, format="%.3f"),
                                    "drifted": st.column_config.CheckboxColumn("Drifted")})
