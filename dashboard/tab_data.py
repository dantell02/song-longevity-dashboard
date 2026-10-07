"""Tab 0: The Data. A warm-up look at the dataset before the analysis results (export_data_overview.py)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import style as S
from .data import load, need, nice, tier_names

GROUP_ORDER = ["audio", "artist", "label", "genre", "distribution", "outcome"]
BINARY = {"editorial_playlist", "featured_artist", "slow_burner", "is_hit"}


def key_cards(meta: dict, counts: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        S.card("Songs", f"{meta['n_songs']:,}", f"{meta['n_columns']} columns each")
    with c2:
        S.card("Genres · label tiers", f"{(counts.variable == 'genre').sum()} · "
               f"{(counts.variable == 'label_tier').sum()}", "categories")
    with c3:
        S.card("A hit means", f"≥ {S.streams(meta['hit_threshold_streams'])} streams",
               f"top {meta['hit_rate']:.0%} by 3-year streams")
    with c4:
        S.card("Missing values", f"{meta['n_missing']:,}", "synthetic dataset, no cleaning needed")


def variable_overview(variables: pd.DataFrame) -> None:
    st.subheader("What we know about each song")
    groups = [g for g in GROUP_ORDER if g in set(variables.group)]
    cols = st.columns(len(groups))
    for col, g in zip(cols, groups):
        items = variables[variables.group == g]
        lines = "\n".join(f"- {nice(r.variable, 'variable')}{' ⏱' if r.post_release else ''}"
                          for r in items.itertuples())
        col.markdown(f"<span style='color:{S.GROUP_COLORS.get(g, '#666')};font-size:1.3rem'>●</span> "
                     f"**{S.GROUP_NAMES.get(g, g)}**\n\n{lines}", unsafe_allow_html=True)
    st.caption("⏱ = only known after release. These are used in the statistical models (TikTok virality) "
               "but excluded from the ML models, which should only use information available before release.")


def preview(rows: pd.DataFrame) -> None:
    st.subheader("What one row looks like")
    d = rows.copy()
    for var in ("genre", "label_tier"):
        d[var] = [nice(v, "value") for v in d[var]]
    for var in BINARY & set(d.columns):
        d[var] = d[var].map({1: "Yes", 0: "No"})
    d = d.rename(columns={c: "Track ID" if c == "track_id" else nice(c, "variable") for c in d.columns})
    st.dataframe(d, hide_index=True, width="stretch")
    st.caption(f"{len(rows)} randomly drawn songs. Audio features are scores from 0 to 10; the marketing budget "
               "is a relative index, not euros.")


def step_hist(h: pd.DataFrame, color: str) -> go.Scatter:
    """Histogram drawn as a filled step line (works on both linear and log x-axes)."""
    x = list(h.bin_left) + [h.bin_right.iloc[-1]]
    y = list(h["count"]) + [h["count"].iloc[-1]]
    return go.Scatter(x=x, y=y, mode="lines", line_shape="hv", fill="tozeroy", line=dict(color=color, width=2),
                      hovertemplate="from %{x:,.2f}<br>%{y:,} songs<extra></extra>", showlegend=False)


def marker_line(x: float, top: float, label: str, color: str, pos: str = "top right") -> go.Scatter:
    """Labelled vertical line as a trace (vlines are unreliable on log axes)."""
    return go.Scatter(x=[x, x], y=[0, top], mode="lines+text", text=["", label], textposition=pos,
                      line=dict(color=color, dash="dash", width=2), textfont=dict(color=color),
                      hoverinfo="skip", showlegend=False, cliponaxis=False)


def streams_distribution(hist: pd.DataFrame, summary: pd.DataFrame, meta: dict) -> None:
    st.subheader("The outcome: 3-year streams")
    h = hist[hist.variable == "streams_3yr"]
    s = summary.set_index("variable").loc["streams_3yr"]
    top = h["count"].max() * 1.1
    fig = go.Figure([step_hist(h, S.GROUP_COLORS["outcome"]),
                     marker_line(s["median"], top, f"Median: {S.streams(s['median'])}",
                                 S.GROUP_COLORS["distribution"], "top left"),
                     marker_line(meta["hit_threshold_streams"], top,
                                 f"Hit: ≥ {S.streams(meta['hit_threshold_streams'])}", S.HIGHLIGHT)])
    fig.update_layout(title="How many streams do songs get in 3 years?",
                      xaxis=dict(type="log", title="3-year streams (log scale)"), yaxis_title="Number of songs")
    S.show(fig, "data_streams", height=450)
    S.caption(f"Most songs get modest numbers (median {S.streams(s['median'])}, middle half between "
              f"{S.streams(s['p25'])} and {S.streams(s['p75'])}), while a few reach {S.streams(s['max'])}. "
              f"The average ({S.streams(s['mean'])}) is pulled far above the median by these outliers.",
              "Streams are extremely skewed, so all models work on the log scale and report effects in %.")


def category_counts(counts: pd.DataFrame) -> None:
    st.subheader("Genres and label tiers")
    c1, c2 = st.columns(2)
    g = counts[counts.variable == "genre"].sort_values("n")
    names = [nice(v, "value") for v in g.value]
    fig = go.Figure(go.Bar(x=g.n, y=names, orientation="h", marker_color=S.GROUP_COLORS["genre"],
                           text=[f"{v:.0%}" for v in g.share], textposition="outside",
                           hovertemplate="%{y}: %{x:,} songs<extra></extra>"))
    fig.update_layout(title="Songs per genre", xaxis_title="Number of songs", xaxis_range=[0, g.n.max() * 1.2])
    with c1:
        S.show(fig, "data_genres", height=420)
    tiers = tier_names()
    t = counts[counts.variable == "label_tier"].set_index("value").reindex(list(tiers)).dropna()
    fig = go.Figure(go.Bar(x=[tiers[k] for k in t.index], y=t.n,
                           marker_color=[S.TIER_COLORS.get(tiers[k], "#666") for k in t.index],
                           text=[f"{v:.0%}" for v in t.share], textposition="outside",
                           hovertemplate="%{x}: %{y:,} songs<extra></extra>"))
    fig.update_layout(title="Songs per label tier", yaxis_title="Number of songs", yaxis_range=[0, t.n.max() * 1.2])
    with c2:
        S.show(fig, "data_tiers", height=420)
    S.caption("How the songs are spread over genres and label tiers.",
              "Every group is large (thousands of songs), so differences between groups are estimated precisely.")


def tier_profile(prof: pd.DataFrame) -> None:
    st.subheader("Not all label tiers start equal")
    tiers = tier_names()
    p = prof.set_index("label_tier").reindex(list(tiers)).dropna()
    names = [tiers[k] for k in p.index]
    colors = [S.TIER_COLORS.get(n, "#666") for n in names]
    panels = [("median_marketing", "Median marketing budget", lambda v: f"{v:.2f}"),
              ("median_listeners", "Median prior listeners", S.streams),
              ("median_streams", "Median 3-year streams", S.streams),
              ("hit_rate", "Share of hits", lambda v: f"{v:.0%}")]
    for col, (var, title, fmt) in zip(st.columns(len(panels)), panels):
        fig = go.Figure(go.Bar(x=names, y=p[var], marker_color=colors, text=[fmt(v) for v in p[var]],
                               textposition="outside", hovertemplate="%{x}: %{text}<extra></extra>"))
        fig.update_layout(title=title, yaxis_range=[0, p[var].max() * 1.25], yaxis_showticklabels=False,
                          margin=dict(l=5, r=5, t=60, b=10))
        with col:
            S.show(fig, f"data_tier_{var}", height=340)
    mk, st_ = p.median_marketing, p.median_streams
    S.caption(f"Major-label songs get a larger marketing budget (median {mk['major']:.2f} vs "
              f"{mk['independent']:.2f} for independents) and far more streams ({S.streams(st_['major'])} vs "
              f"{S.streams(st_['independent'])}), while the artists start out about equally big.",
              "Marketing and label tier move together. To see what marketing itself does, the analysis has to "
              "separate the two; this is exactly what the next tab does.")


def explore_variable(hist: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.subheader("Explore a variable")
    options = [v for v in hist.variable.unique() if v != "streams_3yr"]
    var = st.segmented_control("Variable", options, default=options[0], required=True,
                               format_func=lambda v: nice(v, "variable"), key="data_var")
    h = hist[hist.variable == var]
    s = summary.set_index("variable").loc[var]
    group = load("data_variables.csv")
    g = group.set_index("variable").group.get(var, "outcome") if group is not None else "outcome"
    fig = go.Figure([step_hist(h, S.GROUP_COLORS.get(g, "#666")),
                     marker_line(s["median"], h["count"].max() * 1.1, "Median", S.REF_LINE)])
    fig.update_layout(title=f"Distribution of {nice(var, 'variable')}",
                      xaxis=dict(type="log" if h.scale.iloc[0] == "log" else "linear",
                                 title=nice(var, "variable") + (" (log scale)" if h.scale.iloc[0] == "log" else "")),
                      yaxis_title="Number of songs")
    c1, c2 = st.columns([3, 1])
    with c1:
        S.show(fig, "data_explore", height=420)
    with c2:
        st.metric("Median", f"{s['median']:,.2f}", border=True)
        st.metric("Middle half of songs", f"{s['p25']:,.2f} – {s['p75']:,.2f}", border=True)
        st.metric("Range", f"{s['min']:,.2f} – {s['max']:,.2f}", border=True)
    S.caption("How the selected variable is spread across all songs; the dashed line marks the median.",
              "Audio scores are evenly spread around the middle, while marketing, artist size and TikTok "
              "virality have long tails: a few songs get far more than the rest.")


def render() -> None:
    meta = load("data_metadata.json")
    counts = need("data_counts.csv")
    if meta is None:
        st.error("Missing file `results/data_metadata.json`. Re-run the script **export_data_overview.py** "
                 "to create it.")
    elif counts is not None:
        key_cards(meta, counts)
    variables = need("data_variables.csv")
    if variables is not None:
        variable_overview(variables)
    rows = need("data_preview.csv")
    if rows is not None:
        preview(rows)
    st.divider()
    files = need("data_histograms.csv", "data_summary.csv")
    if files is not None and meta is not None:
        streams_distribution(*files, meta)
    st.divider()
    if counts is not None:
        category_counts(counts)
    st.divider()
    prof = need("data_tier_profile.csv")
    if prof is not None:
        tier_profile(prof)
    st.divider()
    if files is not None:
        explore_variable(*files)
