"""Tab 3: Findings & Where Both Agree. Synthesis of the statistics and ML exports."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import style as S
from .data import load, missing_error, nice, tier_names

STATS_FILES = ["stats_coefficients.csv", "stats_model_fit.csv", "stats_marketing_slopes.csv",
               "stats_genre_hit_prob.csv"]
ML_FILES = ["ml_marginal_marketing_return.csv", "ml_feature_importance.csv", "ml_model_performance.csv",
            "ml_scenario_grid.csv"]
CONTROLS_MODEL, TIKTOK_MODEL, FULL_MODEL = "N3", "N4", "N5"
ML_MODEL = "Linear Regression"

BADGES = {"Agree": "#009E73", "Only statistics": "#0072B2", "Only ML": "#E69F00"}


class Facts:
    """Every number used on this tab, looked up from the exported files."""

    def __init__(self, f: dict[str, pd.DataFrame]):
        c = f["stats_coefficients.csv"]
        n3, n5 = c[c.model_id == CONTROLS_MODEL], c[c.model_id == FULL_MODEL]
        self.mk = n3[n3.term == "marketing_budget"].iloc[0]
        self.labels = n3[n3.term_group == "label"].sort_values("pct_effect", ascending=False)
        self.listeners = n3[n3.term == "log_listeners"].iloc[0]
        audio = n5[n5.term_group == "audio"]
        self.audio_top = audio.loc[audio.pct_effect.abs().idxmax()]

        self.fit = f["stats_model_fit.csv"].set_index("model_id")
        self.slopes = f["stats_marketing_slopes.csv"].query("group_type == 'label_tier'")
        self.hits = f["stats_genre_hit_prob.csv"]

        self.marg = f["ml_marginal_marketing_return.csv"]
        self.ml_mk = self.marg.percent_change.mean()
        imp = f["ml_feature_importance.csv"]
        self.imp = imp[imp.model == ML_MODEL].sort_values("importance", ascending=False)
        self.audio_imp = self.imp[self.imp.group == "audio"].importance.sum()
        self.genre_imp = self.imp[self.imp.group == "genre"].importance.sum()
        self.ml_r2 = f["ml_model_performance.csv"].set_index("Model").R2_Log[ML_MODEL]
        self.grid = f["ml_scenario_grid.csv"]


def agreement_cards(x: Facts) -> None:
    top_label, top_ml = x.labels.iloc[0], x.imp.iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        S.card(f"Marketing: {S.pct(x.mk.pct_effect)} per unit",
               f"{S.pct(x.mk.pct_effect)} | {S.pct(x.ml_mk)}", "Statistics | ML")
    with c2:
        S.card(f"Strongest driver: {nice('label_tier', 'variable')}",
               f"{S.pct(top_label.pct_effect)} | #1 of {len(x.imp)}",
               f"Statistics: {top_label.term_label} vs. independent | ML: {nice(top_ml.feature, 'variable')} "
               f"(importance {top_ml.importance:.3f})")
    with c3:
        S.card("Audio features barely matter", f"{S.pct(x.audio_top.pct_effect)} | {x.audio_imp:.3f}",
               f"Statistics: largest audio effect ({x.audio_top.term_label}) | ML: all audio importances summed")


def badge(verdict: str) -> str:
    return (f"<span style='background:{BADGES[verdict]};color:white;padding:3px 10px;border-radius:12px;"
            f"font-weight:600;white-space:nowrap'>{verdict}</span>")


def agreement_table(x: Facts) -> None:
    st.subheader("Question by question")
    tiers = tier_names()
    top3 = ", ".join(f"{i + 1}. {nice(r.feature, 'variable')} ({r.importance:.3f})"
                     for i, r in enumerate(x.imp.head(3).itertuples()))
    tier_slopes = ", ".join(f"{r.group_label} {S.pct(r.pct_effect)}" for r in x.slopes.itertuples())
    r2 = x.fit.r2
    rows = [
        ("How much does +1 marketing unit add?",
         f"{S.pct(x.mk.pct_effect)} (95% CI {S.ci(x.mk.pct_ci_low, x.mk.pct_ci_high)}, model 3)",
         f"{S.pct(x.ml_mk)} on average per step ({int(x.marg.from_marketing.min())} → "
         f"{int(x.marg.to_marketing.max())})", "Agree"),
        ("What matters most?",
         f"{x.labels.iloc[0].term_label} {S.pct(x.labels.iloc[0].pct_effect)}; prior listeners: +1% → "
         f"{S.pct(x.listeners.estimate)}; marketing {S.pct(x.mk.pct_effect)} per unit",
         top3, "Agree"),
        ("Do audio features matter?",
         f"Largest audio effect: {x.audio_top.term_label} {S.pct(x.audio_top.pct_effect)} (model 5)",
         f"All audio importances together: {x.audio_imp:.3f}", "Agree"),
        ("Does genre matter?",
         f"Hit probability {x.hits.p_hit.min():.1%}–{x.hits.p_hit.max():.1%} per genre vs. "
         f"{x.hits.overall_rate.iloc[0]:.0%} overall",
         f"Genre importance: {x.genre_imp:.4f}", "Agree"),
        ("Does marketing saturate (diminishing returns)?", "Not tested",
         f"{S.pct(x.marg.percent_change.iloc[0])} → {S.pct(x.marg.percent_change.iloc[-1])} per unit: "
         "roughly constant", "Only ML"),
        ("How much does TikTok add?",
         f"R² {r2[CONTROLS_MODEL]:.3f} → {r2[TIKTOK_MODEL]:.3f} when TikTok virality is added",
         "Excluded by design (only known after release)", "Only statistics"),
        (f"Does marketing work differently per {nice('label_tier', 'variable').lower()}?",
         f"{tier_slopes}; joint test p = {S.pval(x.slopes.wald_p.iloc[0])}", "Not tested", "Only statistics"),
    ]
    head = "".join(f"<th style='text-align:left;padding:8px'>{h}</th>"
                   for h in ("Question", "Statistics says", "ML says", "Verdict"))
    body = "".join(
        "<tr style='border-top:1px solid #ddd'>"
        + "".join(f"<td style='padding:8px;vertical-align:top'>{html.escape(c)}</td>" for c in r[:3])
        + f"<td style='padding:8px;vertical-align:top'>{badge(r[3])}</td></tr>"
        for r in rows)
    st.markdown(f"<table style='width:100%;border-collapse:collapse;font-size:1.05rem'>"
                f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", unsafe_allow_html=True)
    st.caption("“Only statistics” or “Only ML” is not a contradiction: it means the question was only asked "
               "by one method (by design, e.g. TikTok is excluded from the ML models), so the other is silent.")


def r2_chart(x: Facts) -> None:
    fit = x.fit.loc[["N1", "N2", "N3", "N4", "N5"]]
    colors = [S.HIGHLIGHT if m == TIKTOK_MODEL else (S.GROUP_COLORS["distribution"] if fit.loc[m, "includes_tiktok"]
              else S.MUTED) for m in fit.index]
    fig = go.Figure(go.Bar(x=fit.model_label, y=fit.r2, marker_color=colors,
                           text=[f"{v:.2f}" for v in fit.r2], textposition="outside",
                           hovertemplate="%{x}<br>R² %{y:.3f}<extra></extra>"))
    fig.add_hline(y=x.ml_r2, line_dash="dash", line_color=S.REF_LINE,
                  annotation_text=f"ML model without TikTok (test set): {x.ml_r2:.2f}",
                  annotation_position="top left")
    fig.update_layout(title="1 · TikTok doubles the explained variance", yaxis_title="R² (share of variance explained)",
                      yaxis_range=[0, fit.r2.max() * 1.25])
    S.show(fig, "headline_r2", height=480)
    S.caption(f"R² of the five statistical models; the highlighted bar is the first model with TikTok virality "
              f"({x.fit.r2[CONTROLS_MODEL]:.2f} → {x.fit.r2[TIKTOK_MODEL]:.2f}). The dashed line is the ML "
              f"linear model, which excludes TikTok.",
              f"This is why the two analyses report different R² values. Without TikTok both sit at about "
              f"{x.fit.r2[CONTROLS_MODEL]:.2f} (statistics) and {x.ml_r2:.2f} (ML), which is itself an agreement.")


def tier_slope_chart(x: Facts) -> None:
    s = x.slopes
    names = [f"{r.group_label} (reference)" if r.is_reference else r.group_label for r in s.itertuples()]
    hover = [f"<b>{n}</b><br>{S.pct(r.pct_effect)} per unit<br>95% CI {S.ci(r.pct_ci_low, r.pct_ci_high)}"
             for n, r in zip(names, s.itertuples())]
    fig = S.dot_ci(names, s.pct_effect, s.pct_ci_low, s.pct_ci_high,
                   [S.TIER_COLORS.get(n, "#666") for n in s.group_label], hover,
                   "2 · Marketing pays off similarly across label tiers",
                   "Change in streams per +1 marketing unit (%)", ref_x=None)
    fig.update_xaxes(rangemode="tozero")
    S.show(fig, "headline_tiers", height=420)
    wp = s.wald_p.iloc[0]
    strength = "weak evidence of a difference, not a finding" if 0.05 <= wp < 0.10 else (
        "evidence of a difference" if wp < 0.05 else "no evidence of a difference")
    S.caption(f"Marketing effect per label tier with 95% CI ({', '.join(S.pct(v) for v in s.pct_effect)}). "
              f"Joint test that the slopes differ: p = {S.pval(wp)}.",
              f"p = {S.pval(wp)} means {strength}: the intervals overlap, so marketing works about equally "
              "for every tier.")


def tier_curves_chart(x: Facts) -> None:
    g = x.grid[(x.grid.listeners_level == "Medium") & (x.grid.energy_level == "Medium")]
    fig = go.Figure()
    for raw, name in reversed(tier_names().items()):
        t = g[g.label_tier == raw].sort_values("marketing_budget")
        fig.add_trace(go.Scatter(x=t.marketing_budget, y=t.predicted_streams, mode="lines", name=name,
                                 line=dict(color=S.TIER_COLORS.get(name, "#666"), width=4),
                                 hovertemplate=f"{name}<br>Budget %{{x:.2f}}<br>%{{y:,.0f}} streams<extra></extra>"))
    fig.update_layout(title="3 · Same budget, different starting point",
                      xaxis_title="Marketing budget (relative index)", yaxis_title="Predicted 3-year streams")
    S.show(fig, "headline_curves", height=480)
    S.caption("ML predictions for a typical song (medium artist size, medium energy), one line per label tier.",
              "Label tier shifts the whole curve up, while marketing lifts every tier at a similar rate: the ML "
              "picture of chart 2.")


def limitations() -> None:
    st.warning(
        "**Limitations**\n"
        "- Synthetic dataset: patterns describe the simulation, not the real music industry.\n"
        "- Associations, not causation: no experiment assigned marketing budgets.\n"
        "- Marketing budget is a relative index (0–12), not euros.\n"
        "- Slow burners (RQ2) are not yet analysed."
    )


def render() -> None:
    st.markdown("#### Two independent methods, one dataset: where do they reach the same conclusion?")
    files = {n: load(n) for n in STATS_FILES + ML_FILES}
    missing = [n for n, f in files.items() if f is None]
    for n in missing:
        missing_error(n)
    if missing:
        st.info("This tab combines both analyses and needs all of the files above.")
    else:
        x = Facts(files)
        agreement_cards(x)
        agreement_table(x)
        st.divider()
        st.subheader("Three headline charts")
        r2_chart(x)
        c1, c2 = st.columns(2)
        with c1:
            tier_slope_chart(x)
        with c2:
            tier_curves_chart(x)
    limitations()
