"""SentinelML dashboard entry point.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="SentinelML · Model Reliability", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

import theme  # noqa: E402

import data  # noqa: E402

theme.apply_theme()
data.ensure_pipeline()

pages = {
    "Monitor": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page("views/live_stream.py", title="Live Stream", icon=":material/stream:", url_path="live_stream"),
        st.Page("views/data_quality.py", title="Data Quality", icon=":material/fact_check:", url_path="data_quality"),
        st.Page("views/drift_events.py", title="Drift Events", icon=":material/ssid_chart:", url_path="drift_events"),
    ],
    "Models": [
        st.Page("views/model_performance.py", title="Model Performance", icon=":material/monitoring:", url_path="model_performance"),
        st.Page("views/explainability.py", title="Explainability", icon=":material/psychology:", url_path="explainability"),
        st.Page("views/model_versions.py", title="Model Versions", icon=":material/inventory_2:", url_path="model_versions"),
    ],
    "Research": [
        st.Page("views/experiment_comparison.py", title="Experiment Comparison", icon=":material/science:", url_path="experiment_comparison"),
    ],
}

with st.sidebar:
    theme.sidebar_brand()

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    r = data.runs()
    theme.status_line(f"EXP #{data.db().latest_experiment_id()} · {len(r)} RUNS · "
                      f"{r['seed'].nunique() if not r.empty else 0} SEEDS")
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        '<div class="disclaimer">Synthetic controlled stream with known drift ground truth. '
        'Research prototype — not a production monitoring system.</div>',
        unsafe_allow_html=True,
    )

nav.run()
theme.footer()
