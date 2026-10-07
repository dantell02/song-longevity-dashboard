"""
Export descriptive summaries of the raw dataset to results/data_*.

The dashboard never reads raw-data/ itself. This script is the only place where the
"The Data" tab's numbers are computed; re-run it if the dataset changes.

Run:  uv run python export_data_overview.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "raw-data" / "song_longevity.csv"
RESULTS = ROOT / "results"

# Variable -> group, in the order the dashboard shows them.
GROUPS = {
    "audio": ["energy", "valence", "danceability", "acousticness", "instrumentalness", "tempo_bpm",
              "song_length_sec"],
    "artist": ["artist_prior_monthly_listeners", "featured_artist"],
    "label": ["label_tier"],
    "distribution": ["marketing_budget", "playlist_adds_first_month", "editorial_playlist", "tiktok_virality"],
    "genre": ["genre"],
    "outcome": ["first_month_streams", "halflife_days", "streams_3yr", "slow_burner", "is_hit"],
}
# Known only after release (excluded from the ML models).
POST_RELEASE = {"playlist_adds_first_month", "editorial_playlist", "tiktok_virality", "first_month_streams"}
LOG_BINNED = {"streams_3yr", "artist_prior_monthly_listeners", "first_month_streams"}
HISTOGRAMS = ["streams_3yr", "marketing_budget", "artist_prior_monthly_listeners", "halflife_days",
              "tiktok_virality", *GROUPS["audio"]]


def variables() -> pd.DataFrame:
    rows = [{"variable": v, "group": g, "post_release": v in POST_RELEASE}
            for g, vs in GROUPS.items() for v in vs]
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    num = [v for vs in GROUPS.values() for v in vs if pd.api.types.is_numeric_dtype(df[v])]
    d = df[num].describe(percentiles=[0.25, 0.5, 0.75]).T
    return (d.rename(columns={"25%": "p25", "50%": "median", "75%": "p75"})
             .reset_index(names="variable")[["variable", "mean", "median", "p25", "p75", "min", "max"]])


def counts(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for var in ("genre", "label_tier"):
        vc = df[var].value_counts()
        out.append(pd.DataFrame({"variable": var, "value": vc.index, "n": vc.values, "share": vc.values / len(df)}))
    return pd.concat(out, ignore_index=True)


def histograms(df: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    out = []
    for var in HISTOGRAMS:
        x = df[var]
        if var in LOG_BINNED:
            edges = np.logspace(np.log10(x[x > 0].min()), np.log10(x.max()), bins + 1)
        else:
            edges = np.linspace(x.min(), x.max(), bins + 1)
        n, edges = np.histogram(x, bins=edges)
        out.append(pd.DataFrame({"variable": var, "scale": "log" if var in LOG_BINNED else "linear",
                                 "bin_left": edges[:-1], "bin_right": edges[1:], "count": n,
                                 "share": n / len(df)}))
    return pd.concat(out, ignore_index=True)


def tier_profile(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("label_tier")
              .agg(n=("track_id", "size"), median_marketing=("marketing_budget", "median"),
                   median_listeners=("artist_prior_monthly_listeners", "median"),
                   median_streams=("streams_3yr", "median"), hit_rate=("is_hit", "mean"))
              .reset_index())


def main() -> None:
    df = pd.read_csv(RAW)
    RESULTS.mkdir(exist_ok=True)
    variables().to_csv(RESULTS / "data_variables.csv", index=False)
    summary(df).to_csv(RESULTS / "data_summary.csv", index=False)
    counts(df).to_csv(RESULTS / "data_counts.csv", index=False)
    histograms(df).to_csv(RESULTS / "data_histograms.csv", index=False)
    tier_profile(df).to_csv(RESULTS / "data_tier_profile.csv", index=False)
    df.sample(8, random_state=42).to_csv(RESULTS / "data_preview.csv", index=False)
    json.dump({
        "script": "export_data_overview.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "raw-data/song_longevity.csv",
        "n_songs": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "hit_threshold_streams": int(df.loc[df.is_hit == 1, "streams_3yr"].min()),
        "slow_burner_halflife_days": 150,
        "hit_rate": float(df.is_hit.mean()),
        "slow_burner_rate": float(df.slow_burner.mean()),
    }, open(RESULTS / "data_metadata.json", "w"), indent=2)
    print("Exported data_* files to", RESULTS)


if __name__ == "__main__":
    main()
