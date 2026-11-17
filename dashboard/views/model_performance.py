import plotly.graph_objects as go
import streamlit as st
import theme as t
from plotly.subplots import make_subplots

import data

t.hero("Model Performance", "Rolling metrics computed when labels arrive, compared against the training baseline. "
       "The red band is the ground-truth concept shift — never visible to the model.", badge="Labels: 1-batch delay", icon="📈")

runs = data.runs()
c1, c2 = st.columns([1, 3])
dt = c1.radio("Drift type", sorted(runs.drift_type.unique()), index=sorted(runs.drift_type.unique()).index("sudden"))
metric = c1.radio("Metric", ["roc_auc", "f1", "accuracy"], format_func=lambda m: {"roc_auc": "ROC-AUC", "f1": "F1", "accuracy": "Accuracy"}[m])
smooth = c1.slider("Smoothing (batches)", 1, 6, 2)
show_band = c1.toggle("Show seed spread", True)

allb = data.all_batch_metrics()
sub = allb[allb.drift_type == dt]
with c2:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    for pol in ["static", "periodic", "drift_aware"]:
        g = sub[sub.policy == pol].copy()
        g["sm"] = g.groupby("seed")[metric].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
        st_ = g.groupby("batch_id")["sm"].agg(["mean", "min", "max"]).reset_index()
        col = t.POLICY_COLORS[pol]
        if show_band:
            fig.add_trace(go.Scatter(x=list(st_.batch_id) + list(st_.batch_id[::-1]),
                                     y=list(st_["max"]) + list(st_["min"][::-1]), fill="toself",
                                     fillcolor=col + "22", line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(x=st_.batch_id, y=st_["mean"], name=pol, line=dict(color=col, width=3, shape="spline")), 1, 1)
    alpha = sub[sub.policy == "static"].groupby("batch_id")["concept_alpha"].mean().reset_index()
    fig.add_trace(go.Scatter(x=alpha.batch_id, y=alpha.concept_alpha, fill="tozeroy", name="concept α",
                             line=dict(color=t.DANGER, width=1), fillcolor="rgba(239,68,68,.18)"), 2, 1)
    fig.update_yaxes(title_text=metric, row=1, col=1)
    fig.update_yaxes(title_text="α", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="batch", row=2, col=1)
    st.plotly_chart(t.style_fig(fig, 470), use_container_width=True)

t.section("Per-run summary", "one row per (policy, seed)")
show = runs[runs.drift_type == dt][["policy", "seed", "mean_auc", "post_drift_auc", "mean_f1", "n_retrains",
                                    "detection_delay", "recovery_batches", "retrain_seconds"]]
st.dataframe(show, use_container_width=True, hide_index=True, column_config={
    "post_drift_auc": st.column_config.ProgressColumn("Post-drift AUC", min_value=0.5, max_value=1.0, format="%.3f"),
    "mean_auc": st.column_config.NumberColumn("Mean AUC", format="%.3f"),
    "mean_f1": st.column_config.NumberColumn("Mean F1", format="%.3f"),
    "retrain_seconds": st.column_config.NumberColumn("Retrain s", format="%.2f"),
})
