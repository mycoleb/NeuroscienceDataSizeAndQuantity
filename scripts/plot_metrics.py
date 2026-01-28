from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "metrics.csv"
OUT_DIR = ROOT / "outputs"


# -----------------------------
# Styling helpers
# -----------------------------
def _apply_theme() -> None:
    """
    Make plots look consistent + modern without extra deps.
    Uses Matplotlib's built-in seaborn-v0_8 style (does NOT require seaborn).
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        # Fallback if style name changes
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _wrap_labels(labels, width: int = 26):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


def _nice_int(x: float) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _nice_float(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return ""
    return f"{x:,.{decimals}f}"


def _add_bar_labels(ax, values, fmt="auto"):
    """
    Add value labels to bars (horizontal).
    fmt:
      - "auto": ints get commas, floats 2dp
      - custom format string like "{:.1f}"
    """
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if fmt == "auto":
            label = _nice_int(v) if float(v).is_integer() else _nice_float(v, 2)
        else:
            label = fmt.format(v)
        ax.text(v, i, f"  {label}", va="center")


# -----------------------------
# Data
# -----------------------------
def _read() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing {DATA_CSV}.\n"
            f"Run your metrics fetch script first (the script that writes data/metrics.csv)."
        )

    df = pd.read_csv(DATA_CSV)

    # Clean numeric columns
    df["dataset_count"] = pd.to_numeric(df.get("dataset_count"), errors="coerce")
    df["total_tb"] = pd.to_numeric(df.get("total_tb"), errors="coerce")

    # Helpful derived columns for nicer labeling/hover logic later
    df["label"] = df["label"].astype(str)
    return df


# -----------------------------
# Plots
# -----------------------------
@dataclass
class PlotConfig:
    top_n: int = 15
    wrap_width: int = 28
    figsize: tuple = (11, 7)


def plot_largest_by_size(df: pd.DataFrame, cfg: PlotConfig = PlotConfig()) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "largest_by_size_tb.png"

    d = df.dropna(subset=["total_tb"]).sort_values("total_tb", ascending=False).head(cfg.top_n).copy()
    if d.empty:
        raise ValueError("No sources have total_tb populated.")

    labels = _wrap_labels(d["label"].tolist(), width=cfg.wrap_width)
    values = d["total_tb"].tolist()

    fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=True)
    ax.barh(labels, values)
    ax.invert_yaxis()

    ax.set_xlabel("Total size (TB)")
    ax.set_title("Largest neuroscience data sources (by published / estimated size)")
    ax.grid(True, axis="x", alpha=0.35)
    ax.grid(False, axis="y")

    _add_bar_labels(ax, values, fmt="auto")

    # Subtitle / footnote-like context
    ax.text(
        0.0,
        -0.10,
        "Note: totals depend on each source’s published metric; some are estimates/fallbacks.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.85,
        ha="left",
        va="top",
    )

    fig.savefig(out)
    plt.close(fig)
    return out


def plot_most_datasets(df: pd.DataFrame, cfg: PlotConfig = PlotConfig()) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "most_datasets.png"

    d = df.dropna(subset=["dataset_count"]).sort_values("dataset_count", ascending=False).head(cfg.top_n).copy()
    if d.empty:
        raise ValueError("No sources have dataset_count populated.")

    labels = _wrap_labels(d["label"].tolist(), width=cfg.wrap_width)
    values = d["dataset_count"].tolist()

    fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=True)
    ax.barh(labels, values)
    ax.invert_yaxis()

    ax.set_xlabel("Dataset / item count (units vary by source)")
    ax.set_title("Neuroscience sources with the most datasets / items")
    ax.grid(True, axis="x", alpha=0.35)
    ax.grid(False, axis="y")

    _add_bar_labels(ax, values, fmt="auto")

    ax.text(
        0.0,
        -0.10,
        "Tip: add a ‘unit’ column in metrics.csv (e.g., datasets, recordings, sessions) for clearer comparisons.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.85,
        ha="left",
        va="top",
    )

    fig.savefig(out)
    plt.close(fig)
    return out


def plot_size_vs_count(df: pd.DataFrame, logx: bool = True, logy: bool = True) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "size_vs_count.png"

    d = df.dropna(subset=["total_tb", "dataset_count"]).copy()
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    if d.empty:
        ax.set_title("Size vs dataset count")
        ax.text(0.5, 0.5, "No sources have BOTH total_tb and dataset_count.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out)
        plt.close(fig)
        return out

    # Scatter
    ax.scatter(d["dataset_count"], d["total_tb"], alpha=0.85)

    # Log scales help readability if values span orders of magnitude
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Dataset count" + (" (log scale)" if logx else ""))
    ax.set_ylabel("Total size (TB)" + (" (log scale)" if logy else ""))
    ax.set_title("Scale comparison: storage vs dataset count")

    ax.grid(True, which="both", alpha=0.25)

    # Simple label placement (kept dependency-free)
    for _, row in d.iterrows():
        x = row["dataset_count"]
        y = row["total_tb"]
        label = row["label"]
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            alpha=0.9,
        )

    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    _apply_theme()
    df = _read()

    cfg = PlotConfig(top_n=15, wrap_width=30, figsize=(12, 7))

    p1 = plot_largest_by_size(df, cfg)
    p2 = plot_most_datasets(df, cfg)
    p3 = plot_size_vs_count(df, logx=True, logy=True)

    print("Wrote:")
    print(f" - {p1}")
    print(f" - {p2}")
    print(f" - {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
