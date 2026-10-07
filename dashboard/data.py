"""Loading the notebook exports from `results/`.

Nothing here computes results: files are read, cached and returned as-is.
A missing file shows an `st.error` naming the file and the notebook that creates it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parent.parent / "results"

STATS_NB = "Jakob_i2pdp.ipynb"
ML_NB = "idil_I2pdp_final(2).ipynb"

# Which notebook creates which file (used in error messages).
DATA_SCRIPT = "export_data_overview.py"

SOURCES = {
    "labels.csv": f"{STATS_NB} / {ML_NB}",
    "data_metadata.json": DATA_SCRIPT,
    "data_variables.csv": DATA_SCRIPT,
    "data_summary.csv": DATA_SCRIPT,
    "data_counts.csv": DATA_SCRIPT,
    "data_histograms.csv": DATA_SCRIPT,
    "data_tier_profile.csv": DATA_SCRIPT,
    "data_preview.csv": DATA_SCRIPT,
    "stats_metadata.json": STATS_NB,
    "stats_coefficients.csv": STATS_NB,
    "stats_model_fit.csv": STATS_NB,
    "stats_marketing_slopes.csv": STATS_NB,
    "stats_genre_hit_prob.csv": STATS_NB,
    "stats_marketing_fit_line.csv": STATS_NB,
    "stats_scatter_sample.csv": STATS_NB,
    "ml_metadata.json": ML_NB,
    "ml_model_performance.csv": ML_NB,
    "ml_feature_importance.csv": ML_NB,
    "ml_marketing_response_curve.csv": ML_NB,
    "ml_marginal_marketing_return.csv": ML_NB,
    "ml_scenario_grid.csv": ML_NB,
    "ml_audio_marketing_interaction_curves.csv": ML_NB,
    "ml_audio_marketing_interactions.csv": ML_NB,
    "ml_test_predictions.csv": ML_NB,
}


@st.cache_data(show_spinner=False)
def _read(name: str, mtime: float):
    """Read one file; `mtime` is part of the cache key so re-run notebooks are picked up."""
    path = RESULTS / name
    if name.endswith(".json"):
        return json.loads(path.read_text())
    return pd.read_csv(path)


def load(name: str):
    """Return the contents of `results/<name>`, or None if the file is missing."""
    path = RESULTS / name
    if not path.exists():
        return None
    return _read(name, path.stat().st_mtime)


def missing_error(name: str) -> None:
    source = SOURCES.get(name, "?")
    kind = "script" if source.endswith(".py") else "notebook"
    st.error(f"Missing file `results/{name}`. Re-run the {kind} **{source}** to create it.")


def need(*names: str):
    """Load several files. Shows an error for each missing one and returns None if any is missing."""
    frames = [load(n) for n in names]
    missing = [n for n, f in zip(names, frames) if f is None]
    for n in missing:
        missing_error(n)
    if missing:
        return None
    return frames[0] if len(frames) == 1 else frames


def metadata(name: str) -> dict:
    """Metadata JSON, or an empty dict when missing (only used for the footer)."""
    return load(name) or {}


# --------------------------------------------------------------------------- #
# Display names from labels.csv
# --------------------------------------------------------------------------- #
def _labels() -> pd.DataFrame:
    df = load("labels.csv")
    return df if df is not None else pd.DataFrame(columns=["kind", "key", "display_name", "group"])


def nice(key: str, kind: str | None = None) -> str:
    """Display name for a raw variable name or category value."""
    lab = _labels()
    rows = lab[lab["key"] == key]
    if kind:
        rows = rows[rows["kind"] == kind]
    if len(rows):
        return str(rows["display_name"].iloc[0])
    return str(key).replace("_", " ").capitalize()


def tier_names() -> dict[str, str]:
    """Raw label tier value -> display name, in a fixed order (smallest to largest)."""
    return {k: nice(k, "value") for k in ("independent", "indie_label", "major")}
