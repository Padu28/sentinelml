import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data
from sentinelml.explainability.explain import explanation_table, global_importance, local_explanation
from sentinelml.ingestion.loaders import load_stream
from sentinelml.retraining.registry import ModelRegistry
from sentinelml.validation.quality import clean_batch

t.hero("Explainability", "Global permutation importance and per-record contributions for the current champion. "
       "Explanations describe model associations — not causal effects.", badge="Champion model", icon="🧠")

s = data.settings
reg = ModelRegistry(data.db(), s.models_dir)
champ = reg.champion()
if champ is None:
    st.warning("No champion model found — run the pipeline first.")
    st.stop()
version, model = champ


@st.cache_data
def recent_data() -> pd.DataFrame:
    df = load_stream("sudden", seed=0)
    return clean_batch(df[df.batch_id >= df.batch_id.max() - 5], s.data["schema"]).reset_index(drop=True)


@st.cache_data
def gi(version: str) -> pd.DataFrame:
    return global_importance(model, recent_data())


rd = recent_data()
c1, c2 = st.columns([1, 1.25])
with c1:
    t.section("Global importance", f"permutation · ROC-AUC drop · model {version}")
    imp = gi(version).sort_values("importance")
    fig = go.Figure(go.Bar(x=imp.importance, y=imp.feature, orientation="h", error_x=dict(array=imp["std"]),
                           marker=dict(color=imp.importance, colorscale=[[0, t.SECONDARY], [1, t.PRIMARY]])))
    st.plotly_chart(t.style_fig(fig, 360), use_container_width=True)

with c2:
    t.section("What-if explorer", "adjust inputs → live prediction & contributions")
    base = rd.iloc[0]
    a, b = st.columns(2)
    rec = {
        "sensor_a": a.slider("sensor_a", -4.0, 5.0, float(round(base.sensor_a, 2)), 0.05),
        "sensor_b": b.slider("sensor_b", -4.0, 4.0, float(round(base.sensor_b, 2)), 0.05),
        "sensor_c": a.slider("sensor_c", -5.0, 5.0, float(round(base.sensor_c, 2)), 0.05),
        "load_kw": b.slider("load_kw", 5.0, 300.0, float(round(base.load_kw, 1)), 0.5),
        "temp_c": a.slider("temp_c", -5.0, 50.0, float(round(base.temp_c, 1)), 0.5),
        "region": b.selectbox("region", ["north", "south", "east", "west"],
                              index=["north", "south", "east", "west"].index(base.region)),
    }
    row = pd.DataFrame([rec])
    expl = local_explanation(model, row, rd)
    p = expl["probability"]
    color = t.DANGER if p >= 0.5 else t.SUCCESS
    st.markdown(f"""<div class="card" style="text-align:center;border-top:3px solid {color}">
        <div style="color:{t.MUTED};font-size:.8rem;font-weight:600;letter-spacing:.06em">FAILURE PROBABILITY</div>
        <div style="font-size:2.6rem;font-weight:800;color:{color}">{p:.1%}</div>
        <div style="color:{t.MUTED};font-size:.8rem">method: {expl['method']}</div></div>""", unsafe_allow_html=True)
    tbl = explanation_table(expl).iloc[::-1]
    fig = go.Figure(go.Waterfall(orientation="h", y=tbl.feature, x=tbl.contribution, connector=dict(line=dict(color=t.BORDER)),
                                 increasing=dict(marker=dict(color=t.DANGER)), decreasing=dict(marker=dict(color=t.SUCCESS))))
    st.plotly_chart(t.style_fig(fig, 300, "Contribution vs. average record"), use_container_width=True)

st.markdown('<div class="disclaimer">Feature contributions are computed relative to the average record in the background '
            'sample. They are associations learned by the model, not causal effects, and can change after retraining.</div>',
            unsafe_allow_html=True)
