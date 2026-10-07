"""Shared colours, number formatting and chart styling (used by every tab)."""
from __future__ import annotations

import math

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Okabe-Ito palette (colour-blind safe), one colour per variable group, used in every tab.
GROUP_COLORS = {
    "distribution": "#0072B2",
    "artist": "#E69F00",
    "label": "#CC79A7",
    "audio": "#009E73",
    "genre": "#56B4E9",
    "interaction": "#D55E00",
    "control": "#999999",
    "outcome": "#444444",
}
GROUP_NAMES = {
    "distribution": "Distribution",
    "artist": "Artist",
    "label": "Label",
    "audio": "Audio",
    "genre": "Genre",
    "interaction": "Interaction",
    "control": "Control",
    "outcome": "Outcome",
}

# Paul Tol "muted" colours for label tiers (colour-blind safe), keyed by display name.
TIER_COLORS = {"Independent": "#88CCEE", "Indie label": "#44AA99", "Major label": "#332288"}

HIGHLIGHT, MUTED, REF_LINE = "#D55E00", "#BBBBBB", "#555555"

pio.templates["dashboard"] = go.layout.Template(
    layout=dict(
        font=dict(size=16),
        title=dict(font=dict(size=22)),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=-0.18, x=0),
        hoverlabel=dict(font_size=15),
    )
)
pio.templates.default = "plotly_white+dashboard"


def show(fig: go.Figure, key: str, height: int = 450) -> None:
    """Render a Plotly figure with the shared layout."""
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", key=key)


def card(label: str, value: str, note: str = "", help: str | None = None) -> None:
    """Key-finding card: big value with a small grey note underneath."""
    st.metric(label, value, delta=note or None, delta_color="off", delta_arrow="off", border=True, help=help)


def dot_ci(labels, est, lo, hi, colors, hover, title: str, x_title: str,
           ref_x: float | None = 0, ref_label: str = "") -> go.Figure:
    """Horizontal dot + 95% CI whisker chart (one row per label)."""
    fig = go.Figure(go.Scatter(
        x=est, y=labels, mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=1, color="white")),
        error_x=dict(type="data", symmetric=False, array=[h - e for h, e in zip(hi, est)],
                     arrayminus=[e - l for e, l in zip(est, lo)], thickness=2.5, width=6, color="#666"),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>", showlegend=False,
    ))
    if ref_x is not None:
        fig.add_vline(x=ref_x, line_dash="dash", line_color=REF_LINE,
                      annotation_text=ref_label, annotation_position="top")
    fig.update_layout(title=title, xaxis_title=x_title)
    return fig


def caption(what: str, why: str) -> None:
    """Standard two-part chart caption."""
    st.caption(f"**What it shows:** {what} **Why it matters:** {why}")


# --------------------------------------------------------------------------- #
# Number formatting
# --------------------------------------------------------------------------- #
def streams(x: float) -> str:
    """51.5K / 1.2M style."""
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.1f}{unit}"
    return f"{x:.0f}"


def pct(x: float) -> str:
    """Signed percentage with one decimal: +33.1%."""
    return f"{x:+.1f}%"


def ci(lo: float, hi: float) -> str:
    return f"{lo:+.1f}% to {hi:+.1f}%"


def pval(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def evidence(p: float) -> str:
    """Plain-language significance badge (emoji + text, so it never relies on colour alone)."""
    if p < 0.001:
        return "🟢 Very strong evidence"
    if p < 0.01:
        return "🟢 Strong"
    if p < 0.05:
        return "🟡 Moderate"
    if p < 0.10:
        return "🟠 Weak"
    return "⚪ No evidence"
