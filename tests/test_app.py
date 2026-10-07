"""Smoke test: the app loads, all three tabs render, and every segmented-control option works.

Run:  uv run pytest -q tests
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TABS = ["The Data", "Statistical Analysis", "Machine Learning Analysis", "Findings & Where Both Agree"]


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_tabs_render():
    at = run_app()
    assert [t.label for t in at.tabs] == TABS
    assert not at.error, [e.value for e in at.error]
    for tab in at.tabs:
        assert len(tab.children) > 0


def test_every_segmented_option():
    at = run_app()
    keys = [sc.key for sc in at.segmented_control]
    assert {"data_var", "stats_model", "ml_imp_model", "sc_tier", "sc_listeners", "sc_energy", "ml_audio_feat"} <= set(keys)
    # Widgets expose only the formatted labels; map them back to raw values via each widget's format_func.
    raw_candidates = ["N1", "N2", "N3", "N4", "N5", "Linear Regression", "Random Forest",
                      "independent", "indie_label", "major", "Low", "Medium", "High",
                      "energy", "instrumentalness", "valence", "danceability", "acousticness", "tempo_bpm",
                      "song_length_sec", "marketing_budget", "artist_prior_monthly_listeners",
                      "halflife_days", "tiktok_virality"]
    checked = 0
    for key in keys:
        sc = at.segmented_control(key=key)
        raw = {str(sc.format_func(c)): c for c in raw_candidates}
        missing = [o for o in sc.options if o not in raw]
        assert not missing, (key, missing)
        for label in sc.options:
            option = raw[label]
            at.segmented_control(key=key).set_value(option).run()
            assert not at.exception, (key, option, at.exception)
            assert not at.error, (key, option, [e.value for e in at.error])
            checked += 1
    at.toggle(key="stats_show_all").set_value(True).run()
    assert not at.exception
    slider = at.select_slider(key="sc_budget")
    for value in (slider.options[0], slider.options[-1]):
        slider.set_value(float(value)).run()
        assert not at.exception
    print(f"checked {checked} segmented-control options across {len(keys)} controls")
