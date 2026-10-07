"""
The Song or the Machine? What Makes a Hit Last
Streamlit dashboard: a presentation layer for the results exported by the two notebooks
(Jakob_i2pdp.ipynb: statistics, idil_I2pdp_final(2).ipynb: machine learning)
and by export_data_overview.py (dataset summaries for "The Data" tab).

The app does no computing of its own. It only reads and visualizes the files in `results/`.

Run:  uv run streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="The Song or the Machine?", page_icon="🎵", layout="wide")

from dashboard import tab_data, tab_findings, tab_ml, tab_stats  # noqa: E402  (after set_page_config)
from dashboard.data import metadata  # noqa: E402

st.title("The Song or the Machine? What Makes a Hit Last")
st.markdown("##### What drives a song's 3-year streams: two methods, statistics and machine learning, on the same songs")
st.info(
    "Synthetic dataset · all results are associations, not causal effects · marketing budget is a relative "
    "index (0–12), not euros · TikTok virality is included in the statistical models but excluded from the "
    "ML models (post-release information)."
)

t0, t1, t2, t3 = st.tabs(["The Data", "Statistical Analysis", "Machine Learning Analysis",
                          "Findings & Where Both Agree"])
with t0:
    tab_data.render()
with t1:
    tab_stats.render()
with t2:
    tab_ml.render()
with t3:
    tab_findings.render()

st.divider()
st.caption(
    f"Results generated: {metadata('data_metadata.json').get('generated_at', 'unknown')} (data overview) / "
    f"{metadata('stats_metadata.json').get('generated_at', 'unknown')} (statistics) / "
    f"{metadata('ml_metadata.json').get('generated_at', 'unknown')} (ML)"
)
