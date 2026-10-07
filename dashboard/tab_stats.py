"""Tab 1: Statistical Analysis (Jakob's notebook). Static charts, readable significance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import style as S
from .data import need

MAIN_MODELS = ["N1", "N2", "N3", "N4", "N5"]
CONTROLS_MODEL = "N3"  # first model controlling for label tier and artist size (no TikTok)


def coef(coefs: pd.DataFrame, model_id: str, term: str) -> pd.Series:
    """One coefficient row."""
    return coefs[(coefs.model_id == model_id) & (coefs.term == term)].iloc[0]


def hover_text(r: pd.Series) -> str:
    return (f"<b>{r.term_label}</b><br>Effect: {S.pct(r.pct_effect)} streams per unit"
            f"<br>95% CI: {S.ci(r.pct_ci_low, r.pct_ci_high)}<br>Std. error: {r.std_error:.3f} (log scale)"
            f"<br>p-value: {S.pval(r.p_value)}")


# --------------------------------------------------------------------------- #
def key_cards(coefs: pd.DataFrame, fit: pd.DataFrame) -> None:
    """Three headline numbers from the statistical models."""
    mk = coef(coefs, CONTROLS_MODEL, "marketing_budget")
    major = coef(coefs, CONTROLS_MODEL, "major")
    r2 = fit.set_index("model_id")["r2"]
    c1, c2, c3 = st.columns(3)
    with c1:
        S.card("Marketing: per +1 budget unit", S.pct(mk.pct_effect) + " streams",
               f"95% CI {S.ci(mk.pct_ci_low, mk.pct_ci_high)} · p {S.pval(mk.p_value)}")
    with c2:
        S.card("Major label vs. independent", S.pct(major.pct_effect) + " streams",
               f"95% CI {S.ci(major.pct_ci_low, major.pct_ci_high)} · p {S.pval(major.p_value)}")
    with c3:
        S.card("Variance explained (R²)", f"{r2['N3']:.2f} → {r2['N4']:.2f}",
               "adding TikTok virality (model 3 → 4)")


def coefficient_section(coefs: pd.DataFrame, fit: pd.DataFrame) -> None:
    """Coefficient plot + results table for one selected model."""
    st.subheader("What is linked to more streams? Effects with 95% confidence intervals")
    labels = coefs.drop_duplicates("model_id").set_index("model_id")["model_label"]
    c1, c2 = st.columns([3, 1])
    with c1:
        model = st.segmented_control("Model", MAIN_MODELS, default=CONTROLS_MODEL, required=True,
                                     format_func=lambda m: labels.get(m, m), key="stats_model")
    with c2:
        show_all = st.toggle("Show genre and audio terms", value=False, key="stats_show_all")

    m = coefs[coefs.model_id == model]
    plot = m[m.pct_effect.notna()]
    if not show_all:
        plot = plot[~plot.term_group.isin(["genre", "audio"])]
    plot = plot.sort_values("pct_effect")

    fig = go.Figure()
    for group, g in plot.groupby("term_group", sort=False):
        fig.add_trace(go.Scatter(
            x=g.pct_effect, y=g.term_label, mode="markers", name=S.GROUP_NAMES.get(group, group),
            marker=dict(size=14, color=S.GROUP_COLORS.get(group, "#666")),
            error_x=dict(type="data", symmetric=False, array=g.pct_ci_high - g.pct_effect,
                         arrayminus=g.pct_effect - g.pct_ci_low, thickness=2.5, width=6),
            customdata=[hover_text(r) for r in g.itertuples()], hovertemplate="%{customdata}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color=S.REF_LINE)
    fig.update_layout(title=f"Effect on 3-year streams · {labels[model]}",
                      xaxis_title="Change in 3-year streams (%)",
                      yaxis=dict(categoryorder="array", categoryarray=list(plot.term_label)))
    S.show(fig, "coef_plot", height=max(320, 70 + 45 * len(plot)))

    listeners = m[m.term == "log_listeners"]
    lis_txt = (f" Artist size is an elasticity and is not plotted: +1% prior listeners → "
               f"{S.pct(listeners.estimate.iloc[0])} streams." if len(listeners) else "")
    S.caption("Each dot is the estimated % change in 3-year streams for a one-unit increase (or versus the "
              "reference group: independent artists, pop). The whisker is the 95% confidence interval; "
              "whiskers that do not cross the dashed 0% line are statistically significant." + lis_txt,
              "It shows both the size of each effect and how certain we are about it, in one picture.")

    results_table(m, fit, model)


def results_table(m: pd.DataFrame, fit: pd.DataFrame, model: str) -> None:
    """Readable table of the selected model (same data as the plot)."""
    rows = []
    for r in m[m.term != "Intercept"].itertuples():
        if pd.isna(r.pct_effect):  # log_listeners: elasticity
            effect = f"+1% listeners → {S.pct(r.estimate)} streams"
            ci = f"{S.pct(r.ci_low)} to {S.pct(r.ci_high)} (per +1%)"
        else:
            effect, ci = S.pct(r.pct_effect), S.ci(r.pct_ci_low, r.pct_ci_high)
        rows.append({"Variable": r.term_label, "Effect on streams": effect, "95% CI": ci,
                     "Std. error (log scale)": f"{r.std_error:.3f}", "p-value": S.pval(r.p_value),
                     "Significance": S.evidence(r.p_value)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    f = fit[fit.model_id == model].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (in-sample)", f"{f.r2:.3f}")
    c2.metric("Adjusted R²", f"{f.adj_r2:.3f}")
    c3.metric("Songs (N)", f"{int(f.n):,}")


def marketing_across_models(coefs: pd.DataFrame) -> None:
    """Marketing effect as controls are added (N1 → N5)."""
    st.subheader("How the marketing effect changes as controls are added")
    mk = coefs[(coefs.term == "marketing_budget") & coefs.model_id.isin(MAIN_MODELS)].copy()
    mk["order"] = mk.model_id.map(MAIN_MODELS.index)
    mk = mk.sort_values("order")
    fig = go.Figure(go.Scatter(
        x=mk.model_label, y=mk.pct_effect, mode="lines+markers+text",
        text=[S.pct(v) for v in mk.pct_effect], textposition="top center",
        marker=dict(size=14, color=S.GROUP_COLORS["distribution"]),
        line=dict(color=S.GROUP_COLORS["distribution"], dash="dot"),
        error_y=dict(type="data", symmetric=False, array=mk.pct_ci_high - mk.pct_effect,
                     arrayminus=mk.pct_effect - mk.pct_ci_low, thickness=2.5, width=8),
        customdata=[hover_text(r) for r in mk.itertuples()], hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(title="Marketing effect per +1 budget unit, model by model",
                      yaxis_title="Change in 3-year streams (%)", yaxis_rangemode="tozero")
    S.show(fig, "mk_models", height=420)
    first, ctrl, last = (mk.set_index("model_id").pct_effect[k] for k in ("N1", CONTROLS_MODEL, "N5"))
    S.caption(f"Without controls one more marketing unit goes with {S.pct(first)} streams; once label tier and "
              f"artist size are controlled for it drops to {S.pct(ctrl)} and stays stable ({S.pct(last)} in the "
              "full model).",
              "Big labels and big artists also spend more on marketing, so part of the raw effect was really "
              "theirs (confounding). The stable value after controls is the more trustworthy estimate.")


def regression_graph(coefs: pd.DataFrame, line: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Scatter of sampled songs with the two fitted lines."""
    st.subheader("The regression line, with and without controls")
    no_c = coef(coefs, "N1", "marketing_budget")
    ctrl = coef(coefs, CONTROLS_MODEL, "marketing_budget")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=sample.marketing_budget, y=sample.log_streams, mode="markers",
                               name=f"Songs (sample of {len(sample):,})",
                               marker=dict(color="#9a9a9a", size=5, opacity=0.25),
                               hovertemplate="Budget %{x:.2f}<br>log streams %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=line.marketing_budget, y=line.fit_no_controls, mode="lines",
                             name=f"No controls: {S.pct(no_c.pct_effect)} per unit",
                             line=dict(color=S.HIGHLIGHT, width=4, dash="dash")))
    fig.add_trace(go.Scatter(x=line.marketing_budget, y=line.fit_controls, mode="lines",
                             name=f"With controls: {S.pct(ctrl.pct_effect)} per unit",
                             line=dict(color=S.GROUP_COLORS["distribution"], width=4)))
    fig.update_layout(title="3-year streams vs. marketing budget",
                      xaxis_title="Marketing budget (relative index)", yaxis_title="3-year streams (log scale)")
    S.show(fig, "reg_graph", height=500)
    S.caption("Grey dots are individual songs; the lines are the fitted regressions without and with controls "
              "(label tier, artist size, genre). The y-axis is log streams, so a straight line means a constant "
              "% gain per unit.",
              "Songs vary enormously around the line: marketing shifts the average, it does not guarantee a hit.")


def group_slopes(slopes: pd.DataFrame) -> None:
    """Marketing effect per label tier and per genre."""
    st.subheader("Does marketing work differently for different groups?")
    c1, c2 = st.columns(2)
    for col, gtype, title in [(c1, "label_tier", "By label tier"), (c2, "genre", "By genre")]:
        g = slopes[slopes.group_type == gtype]
        if gtype == "genre":
            g = g.sort_values("pct_effect")
            colors = [S.GROUP_COLORS["genre"]] * len(g)
        else:
            colors = [S.TIER_COLORS.get(n, "#666") for n in g.group_label]
        names = [f"{r.group_label} (reference)" if r.is_reference else r.group_label for r in g.itertuples()]
        hover = [f"<b>{n}</b><br>Effect: {S.pct(r.pct_effect)} per unit<br>95% CI: "
                 f"{S.ci(r.pct_ci_low, r.pct_ci_high)}<br>p-value: {S.pval(r.p_value)}"
                 for n, r in zip(names, g.itertuples())]
        fig = S.dot_ci(names, g.pct_effect, g.pct_ci_low, g.pct_ci_high, colors, hover,
                       title, "Change in streams per +1 marketing unit (%)", ref_x=None)
        fig.update_xaxes(rangemode="tozero")
        with col:
            S.show(fig, f"slopes_{gtype}", height=420)
            S.caption(f"Marketing effect estimated separately per group. Joint test that the slopes differ: "
                      f"p = {S.pval(g.wald_p.iloc[0])}.",
                      "Overlapping intervals and a non-significant joint test mean marketing works about "
                      "equally well across groups.")


def hit_probability(hits: pd.DataFrame) -> None:
    """Probability of a hit per genre vs. the overall rate."""
    st.subheader("Does genre decide whether a song becomes a hit?")
    h = hits.sort_values("p_hit")
    overall = h.overall_rate.iloc[0]
    hover = [f"<b>{r.genre_label}</b><br>Hit probability: {r.p_hit:.1%}<br>95% CI: {r.ci_low:.1%} to "
             f"{r.ci_high:.1%}" for r in h.itertuples()]
    fig = S.dot_ci(h.genre_label, h.p_hit * 100, h.ci_low * 100, h.ci_high * 100,
                   [S.GROUP_COLORS["genre"]] * len(h), hover, "Chance of becoming a hit (top 10%), by genre",
                   "Hit probability (%)", ref_x=overall * 100, ref_label=f"Overall: {overall:.0%}")
    S.show(fig, "hit_prob", height=430)
    S.caption(f"Share of songs per genre that become a hit, with 95% CI; the dashed line is the overall rate "
              f"({overall:.0%}). Genres range from {h.p_hit.min():.1%} to {h.p_hit.max():.1%}.",
              "Genre barely matters: every genre sits close to the overall rate.")


def glossary() -> None:
    with st.expander("Glossary: how to read these numbers"):
        st.markdown(
            "- **p-value**: how surprising the data would be if there were truly no effect. Small (< 0.05) = "
            "unlikely to be chance. *Like a smoke alarm: the lower the p-value, the less likely it is a false "
            "alarm.*\n"
            "- **Confidence interval (95% CI)**: the range of plausible values for the effect. *Like a weather "
            "forecast saying \"18–22 °C\" rather than a single number.*\n"
            "- **Standard error**: how much the estimate would wobble if we collected the data again. *Like "
            "the spread of your times when you run the same route several times.*\n"
            "- **R²**: share of the differences in streams the model explains (0 = nothing, 1 = everything). "
            "*Like knowing a few ingredients of a recipe: you can guess the taste, but not perfectly.*\n"
            "- **Robust (HC3) standard errors**: standard errors that stay valid when some songs are much more "
            "unpredictable than others. *Like a tolerance that is wide enough for both calm and windy days.*\n"
            "- **Log scale**: each step is a multiplication, not an addition, so effects read as percentages. "
            "*Like the Richter scale for earthquakes.*"
        )


# --------------------------------------------------------------------------- #
def render() -> None:
    files = need("stats_coefficients.csv", "stats_model_fit.csv")
    if files is not None:
        coefs, fit = files
        key_cards(coefs, fit)
        coefficient_section(coefs, fit)
        st.divider()
        marketing_across_models(coefs)
        st.divider()
        more = need("stats_marketing_fit_line.csv", "stats_scatter_sample.csv")
        if more is not None:
            regression_graph(coefs, *more)
        st.divider()

    slopes = need("stats_marketing_slopes.csv")
    if slopes is not None:
        group_slopes(slopes)
    st.divider()
    hits = need("stats_genre_hit_prob.csv")
    if hits is not None:
        hit_probability(hits)
    glossary()
