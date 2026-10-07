"""Tab 2: Machine Learning Analysis (Idil's notebook). Interactive lookups into exported predictions."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import style as S
from .data import need, nice, tier_names

SIMPLE_MODELS = ["Linear Regression", "Polynomial Regression"]
LEVELS = ["Low", "Medium", "High"]
LEVEL_STYLE = {"Low": ("#9ecae1", "dot"), "Medium": ("#4292c6", "dash"), "High": ("#08306b", "solid")}


def key_cards(perf: pd.DataFrame) -> None:
    best = perf.loc[perf.R2_Log.idxmax()]
    linear = perf.set_index("Model").R2_Log.get("Linear Regression")
    flexible = perf[~perf.Model.isin(SIMPLE_MODELS)]
    best_flex = flexible.loc[flexible.R2_Log.idxmax()] if len(flexible) else None
    c1, c2, c3 = st.columns(3)
    with c1:
        S.card("Best test R²", f"{best.R2_Log:.3f}", best.Model)
    with c2:
        S.card("Models compared", f"{len(perf)}", ", ".join(perf.Model))
    with c3:
        if best_flex is not None and linear is not None:
            S.card("Simple beats complex", f"{linear:.3f} vs {best_flex.R2_Log:.3f}",
                   f"Linear regression vs. best flexible model ({best_flex.Model})")


def leaderboard(perf: pd.DataFrame) -> None:
    st.subheader("Model leaderboard: does a more flexible model predict better?")
    p = perf.sort_values("R2_Log")
    colors = [S.GROUP_COLORS["distribution"] if m in SIMPLE_MODELS else S.MUTED for m in p.Model]
    fig = go.Figure(go.Bar(x=p.R2_Log, y=p.Model, orientation="h", marker_color=colors,
                           text=[f"{v:.3f}" for v in p.R2_Log], textposition="outside",
                           hovertemplate="%{y}<br>Test R²: %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Test-set R² (higher is better)", xaxis_title="R² on held-out test songs",
                      xaxis_range=[0, p.R2_Log.max() * 1.25])
    c1, c2 = st.columns([3, 2])
    with c1:
        S.show(fig, "leaderboard", height=360)
    with c2:
        st.dataframe(perf.rename(columns={"R2_Log": "R²", "MAE_Log": "MAE (log)", "RMSE_Log": "RMSE (log)"})
                     .style.format({"R²": "{:.3f}", "MAE (log)": "{:.3f}", "RMSE (log)": "{:.3f}"}),
                     hide_index=True, width="stretch")
    r = perf.set_index("Model").R2_Log
    same = ""
    if set(SIMPLE_MODELS) <= set(r.index):
        same = (f" Linear and polynomial regression score the same ({r['Linear Regression']:.3f} vs "
                f"{r['Polynomial Regression']:.3f}): the extra curve adds nothing.")
    S.caption("Each model was trained on 80% of the songs and scored on the 20% it never saw." + same,
              "The flexible models (random forest, gradient boosting) do not beat a straight line, so the "
              "relationships in the data are essentially simple.")


def feature_importance(imp: pd.DataFrame) -> None:
    st.subheader("Which features does the model rely on?")
    models = list(imp.model.unique())
    model = st.segmented_control("Model", models, default=models[0], required=True, key="ml_imp_model")
    d = imp[imp.model == model].sort_values("importance").copy()
    d["name"] = [nice(f, "variable") for f in d.feature]
    fig = go.Figure()
    for group, g in d.groupby("group", sort=False):
        fig.add_trace(go.Bar(x=g.importance, y=g.name, orientation="h", name=S.GROUP_NAMES.get(group, group),
                             marker_color=S.GROUP_COLORS.get(group, "#666"),
                             hovertemplate="%{y}<br>Drop in R² when shuffled: %{x:.4f}<extra></extra>"))
    fig.update_layout(title=f"Permutation importance · {model}", xaxis_title="Drop in test R² when shuffled",
                      yaxis=dict(categoryorder="array", categoryarray=list(d.name)), barmode="relative")
    S.show(fig, "importance", height=520)
    S.caption("How much the model's accuracy drops when one feature is randomly shuffled. Values at or "
              "slightly below zero mean the feature carries no useful information.",
              "Label tier, artist size and marketing carry the model; the audio features barely matter.")


def scenario_explorer(grid: pd.DataFrame) -> None:
    st.subheader("Scenario explorer: predicted streams for a release")
    tiers = tier_names()
    lis = grid.drop_duplicates("listeners_level").set_index("listeners_level").listeners_value
    en = grid.drop_duplicates("energy_level").set_index("energy_level").energy_value
    c1, c2, c3 = st.columns(3)
    with c1:
        tier = st.segmented_control("Label tier", list(tiers), default="independent", required=True,
                                    format_func=tiers.get, key="sc_tier")
    with c2:
        listeners = st.segmented_control(
            "Artist's prior listeners", LEVELS, default="Medium", required=True, key="sc_listeners",
            help=" · ".join(f"{lv}: {lis[lv]:,.0f} monthly listeners" for lv in LEVELS if lv in lis))
    with c3:
        energy = st.segmented_control(
            "Energy", LEVELS, default="Medium", required=True, key="sc_energy",
            help=" · ".join(f"{lv}: {en[lv]:.2f}" for lv in LEVELS if lv in en) + " (25th / 50th / 75th percentile)")
    budgets = sorted(grid.marketing_budget.unique())
    budget = st.select_slider("Marketing budget (relative index)", budgets, value=budgets[len(budgets) // 2],
                              format_func=lambda b: f"{b:.2f}", key="sc_budget")

    def curve(t, l, e):
        return grid[(grid.label_tier == t) & (grid.listeners_level == l)
                    & (grid.energy_level == e)].sort_values("marketing_budget")

    sel = curve(tier, listeners, energy)
    ref = curve("independent", "Medium", "Medium")
    point = sel[sel.marketing_budget == budget].iloc[0]
    color = S.TIER_COLORS.get(tiers[tier], "#666")

    fig = go.Figure()
    if (tier, listeners, energy) != ("independent", "Medium", "Medium"):
        fig.add_trace(go.Scatter(x=ref.marketing_budget, y=ref.predicted_streams, mode="lines",
                                 name="Reference: Independent · Medium listeners · Medium energy",
                                 line=dict(color=S.MUTED, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=sel.marketing_budget, y=sel.predicted_streams, mode="lines",
                             name=f"Your scenario: {tiers[tier]} · {listeners} listeners · {energy} energy",
                             line=dict(color=color, width=4)))
    fig.add_trace(go.Scatter(x=[budget], y=[point.predicted_streams], mode="markers",
                             name=f"Selected budget ({budget:.2f})",
                             marker=dict(size=18, color=S.HIGHLIGHT, symbol="diamond",
                                         line=dict(width=2, color="white"))))
    fig.update_layout(title="Predicted 3-year streams vs. marketing budget",
                      xaxis_title="Marketing budget (relative index)", yaxis_title="Predicted 3-year streams")
    c1, c2 = st.columns([3, 1])
    with c1:
        S.show(fig, "scenario", height=480)
    with c2:
        st.metric("Predicted 3-year streams", S.streams(point.predicted_streams), border=True,
                  help=f"At marketing budget {budget:.2f}")
        st.metric("Reference scenario at same budget",
                  S.streams(ref[ref.marketing_budget == budget].predicted_streams.iloc[0]), border=True)
    S.caption("A lookup into the model's exported predictions: pick a label tier, artist size and energy level, "
              "and the curve shows predicted streams across the marketing range (other features at typical "
              "values).",
              "It turns the model into a concrete what-if: a typical song, not a guarantee for any single song.")


def diminishing_returns(curve: pd.DataFrame, marg: pd.DataFrame) -> None:
    st.subheader("Does marketing stop paying off?")
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=curve.marketing_budget, y=curve.predicted_streams_3yr, mode="lines",
                               line=dict(color=S.GROUP_COLORS["distribution"], width=4),
                               hovertemplate="Budget %{x:.2f}<br>%{y:,.0f} streams<extra></extra>"))
    fig.update_layout(title="Response curve", xaxis_title="Marketing budget (relative index)",
                      yaxis_title="Predicted 3-year streams")
    with c1:
        S.show(fig, "resp_curve", height=420)
    steps = [f"{int(a)} → {int(b)}" for a, b in zip(marg.from_marketing, marg.to_marketing)]
    fig = go.Figure(go.Bar(x=steps, y=marg.percent_change, marker_color=S.GROUP_COLORS["distribution"],
                           text=[S.pct(v) for v in marg.percent_change], textposition="outside",
                           customdata=[S.streams(v) for v in marg.absolute_change],
                           hovertemplate="Budget %{x}<br>%{y:.1f}% (+%{customdata} streams)<extra></extra>"))
    fig.update_layout(title="Gain per +1 budget unit", xaxis_title="Marketing budget step",
                      yaxis_title="Change in predicted streams (%)",
                      yaxis_range=[0, marg.percent_change.max() * 1.25])
    with c2:
        S.show(fig, "marg_return", height=420)
    S.caption(f"Left: predicted streams rise steadily with marketing. Right: each extra unit adds "
              f"{S.pct(marg.percent_change.max())} to {S.pct(marg.percent_change.min())}, almost the same "
              "percentage at every step.",
              "A constant % gain means no evidence of diminishing returns within the observed budget range.")


def audio_interaction(curves: pd.DataFrame, coefs: pd.DataFrame | None) -> None:
    st.subheader("Does the sound change how well marketing works?")
    feats = list(curves.audio_feature.unique())
    feat = st.segmented_control("Audio feature", feats, default=feats[0], required=True,
                                format_func=lambda f: nice(f, "variable"), key="ml_audio_feat")
    d = curves[curves.audio_feature == feat]
    fig = go.Figure()
    for level, g in d.groupby("audio_level", sort=False):
        short = level.split(" ")[0]
        col, dash = LEVEL_STYLE.get(short, ("#666", "solid"))
        fig.add_trace(go.Scatter(x=g.marketing_budget, y=g.predicted_streams_3yr, mode="lines",
                                 name=f"{level}: {g.audio_value.iloc[0]:.2f}",
                                 line=dict(color=col, dash=dash, width=3)))
    fig.update_layout(title=f"Predicted streams by marketing, for low / medium / high {nice(feat, 'variable')}",
                      xaxis_title="Marketing budget (relative index)", yaxis_title="Predicted 3-year streams")
    S.show(fig, "audio_inter", height=450)
    coef_txt = ""
    if coefs is not None:
        coef_txt = " Interaction coefficients (log scale): " + "; ".join(
            f"{nice(r.Feature, 'variable')} {r.Coefficient:+.4f}" for r in coefs.itertuples()) + "."
    S.caption("Three curves, one per level of the selected audio feature. Parallel curves mean the sound "
              "does not change how strongly marketing works." + coef_txt,
              "The interactions are tiny: marketing works the same way regardless of how a song sounds.")


def fit_check(pred: pd.DataFrame) -> None:
    st.subheader("Model fit check: how close are the predictions?")
    c1, c2 = st.columns(2)
    lo, hi = min(pred.actual_log.min(), pred.predicted_log.min()), max(pred.actual_log.max(), pred.predicted_log.max())
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=pred.actual_log, y=pred.predicted_log, mode="markers", name="Test songs",
                               marker=dict(size=4, color=S.GROUP_COLORS["distribution"], opacity=0.15),
                               hovertemplate="Actual %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=S.HIGHLIGHT, dash="dash", width=3)))
    fig.update_layout(title="Actual vs. predicted (test set)", xaxis_title="Actual 3-year streams (log)",
                      yaxis_title="Predicted 3-year streams (log)")
    with c1:
        S.show(fig, "act_pred", height=450)
    fig = go.Figure(go.Histogram(x=pred.residual, nbinsx=60, marker_color=S.GROUP_COLORS["distribution"]))
    fig.add_vline(x=0, line_dash="dash", line_color=S.REF_LINE)
    fig.update_layout(title="Prediction errors (residuals)", xaxis_title="Actual − predicted (log streams)",
                      yaxis_title="Number of test songs")
    with c2:
        S.show(fig, "residuals", height=450)
    S.caption(f"Left: each of the {len(pred):,} test songs; points on the dashed line would be perfect. "
              "The cloud is much flatter than the line. Right: distribution of the errors.",
              "The model pulls extreme hits and flops toward the average: good for typical songs, poor at "
              "spotting outliers.")


def render() -> None:
    perf = need("ml_model_performance.csv")
    if perf is not None:
        key_cards(perf)
        leaderboard(perf)
    st.divider()
    imp = need("ml_feature_importance.csv")
    if imp is not None:
        feature_importance(imp)
    st.divider()
    grid = need("ml_scenario_grid.csv")
    if grid is not None:
        scenario_explorer(grid)
    st.divider()
    files = need("ml_marketing_response_curve.csv", "ml_marginal_marketing_return.csv")
    if files is not None:
        diminishing_returns(*files)
    st.divider()
    curves = need("ml_audio_marketing_interaction_curves.csv")
    if curves is not None:
        audio_interaction(curves, need("ml_audio_marketing_interactions.csv"))
    st.divider()
    pred = need("ml_test_predictions.csv")
    if pred is not None:
        fit_check(pred)
