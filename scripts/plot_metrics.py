import os
import pandas as pd
import numpy as np

def bilingual(en: str, fr: str) -> str:
    """Stacked English / French text for titles & labels."""
    return f"{en}\n{fr}"

# Create directory structure
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Create a sample metrics.csv with 2026-aligned data
data = {
    'label': [
        'DANDI Archive', 
        'Allen Institute (BKP)', 
        'OpenNeuro', 
        'Human Connectome Project (HCP)', 
        'NeuroMorpho.org',
        'EBRAINS (Human Brain Project)',
        'CRCNS',
        'Mouse Light (Janelia)'
    ],
    'total_tb': [893.0, 500.0, 85.0, 62.0, 0.5, 45.0, 12.0, 150.0],
    'dataset_count': [1023, 50, 1100, 1200, 260078, 200, 150, 40],
    'modality': [
        'Electrophysiology', 
        'Microscopy/Imaging', 
        'MRI/fMRI', 
        'MRI/fMRI', 
        'Morphology', 
        'Multi-modal', 
        'Electrophysiology', 
        'Microscopy/Imaging'
    ],
    'species': ['Multiple', 'Human/Mouse', 'Human', 'Human', 'Multiple', 'Human/Rat', 'Multiple', 'Mouse']
}

df = pd.DataFrame(data)
df.to_csv('data/metrics.csv', index=False)

# Now define the improved plotting script logic
import matplotlib.pyplot as plt
import textwrap

def apply_enhanced_theme():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfb",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def wrap_labels(labels, width=20):
    return [textwrap.fill(str(label), width=width) for label in labels]

def plot_largest_by_size_v2(df):
    d = df.sort_values("total_tb", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a color palette
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(d)))
    bars = ax.barh(wrap_labels(d['label']), d['total_tb'], color=colors, edgecolor='white', height=0.7)
    
    ax.set_xlabel(
    bilingual(
        "Storage Volume (Terabytes)",
        "Volume de stockage (téraoctets)"
    ),
    fontsize=12
    )

    ax.set_title(
        bilingual(
            "The Titans of Neuroscience Data (2026)",
            "Les géants des données en neurosciences (2026)"
        ),
        pad=20
    )

    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{width:,.1f} TB', 
                va='center', fontweight='bold', color='#444444')
    
    plt.tight_layout()
    plt.savefig('outputs/largest_by_size_v2.png')

def plot_modality_pie(df):
    modality_data = df.groupby('modality')['total_tb'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Pastel2(np.linspace(0, 1, len(modality_data)))
    
    wedges, texts, autotexts = ax.pie(
        modality_data, 
        labels=modality_data.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85, 
        explode=[0.05]*len(modality_data),
        textprops={'fontsize': 10}
    )
    
    # Doughnut hole
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    plt.setp(autotexts, size=9, weight="bold", color="black")
    ax.set_title(
        bilingual(
            "Neuroscience Data Landscape by Modality\n(Share of Total TB)",
            "Panorama des données en neurosciences par modalité\n(Part du volume total en To)"
        ),
        fontsize=14,
        pad=10
)

    plt.tight_layout()
    plt.savefig('outputs/modality_distribution_pie.png')

def plot_count_vs_size_bubble(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Size the bubbles by TB
    # Use a log scale for counts because NeuroMorpho is huge
    scatter = ax.scatter(
        df['dataset_count'], 
        df['total_tb'], 
        s=df['total_tb']*2, # Bubble size linked to TB
        c=np.arange(len(df)), 
        cmap='plasma', 
        alpha=0.6, 
        edgecolors="white"
    )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(
        bilingual(
            "Number of Datasets / Items (Log Scale)",
            "Nombre de jeux de données / éléments (échelle logarithmique)"
        ),
        fontsize=12
    )

    ax.set_ylabel(
        bilingual(
            "Total Size in TB (Log Scale)",
            "Taille totale en To (échelle logarithmique)"
        ),
        fontsize=12
    )

    ax.set_title(
        bilingual(
            "Repository Scale: Count vs. Volume",
            "Échelle des dépôts : nombre vs volume"
        ),
        pad=20
    )

    for i, txt in enumerate(df['label']):
        ax.annotate(txt, (df['dataset_count'][i], df['total_tb'][i]), 
                    xytext=(5,5), textcoords='offset points', fontsize=9)

    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.tight_layout()
    plt.savefig('outputs/count_vs_size_bubble.png')

apply_enhanced_theme()
plot_largest_by_size_v2(df)
plot_modality_pie(df)
plot_count_vs_size_bubble(df)

print("Plots generated in outputs/")
# from __future__ import annotations

# import math
# import textwrap
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional

# import matplotlib.pyplot as plt
# import pandas as pd


# ROOT = Path(__file__).resolve().parents[1]
# DATA_CSV = ROOT / "data" / "metrics.csv"
# OUT_DIR = ROOT / "outputs"


# # -----------------------------
# # Styling helpers
# # -----------------------------
# def _apply_theme() -> None:
#     """
#     Make plots look consistent + modern without extra deps.
#     Uses Matplotlib's built-in seaborn-v0_8 style (does NOT require seaborn).
#     """
#     try:
#         plt.style.use("seaborn-v0_8-whitegrid")
#     except Exception:
#         # Fallback if style name changes
#         plt.style.use("default")

#     plt.rcParams.update(
#         {
#             "figure.dpi": 150,
#             "savefig.dpi": 220,
#             "font.size": 11,
#             "axes.titlesize": 14,
#             "axes.labelsize": 11,
#             "xtick.labelsize": 10,
#             "ytick.labelsize": 10,
#             "legend.fontsize": 10,
#             "axes.spines.top": False,
#             "axes.spines.right": False,
#         }
#     )


# def _wrap_labels(labels, width: int = 26):
#     return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


# def _nice_int(x: float) -> str:
#     try:
#         return f"{int(x):,}"
#     except Exception:
#         return str(x)


# def _nice_float(x: float, decimals: int = 2) -> str:
#     if x is None or (isinstance(x, float) and (math.isnan(x))):
#         return ""
#     return f"{x:,.{decimals}f}"


# def _add_bar_labels(ax, values, fmt="auto"):
#     """
#     Add value labels to bars (horizontal).
#     fmt:
#       - "auto": ints get commas, floats 2dp
#       - custom format string like "{:.1f}"
#     """
#     for i, v in enumerate(values):
#         if v is None or (isinstance(v, float) and math.isnan(v)):
#             continue
#         if fmt == "auto":
#             label = _nice_int(v) if float(v).is_integer() else _nice_float(v, 2)
#         else:
#             label = fmt.format(v)
#         ax.text(v, i, f"  {label}", va="center")


# # -----------------------------
# # Data
# # -----------------------------
# def _read() -> pd.DataFrame:
#     if not DATA_CSV.exists():
#         raise FileNotFoundError(
#             f"Missing {DATA_CSV}.\n"
#             f"Run your metrics fetch script first (the script that writes data/metrics.csv)."
#         )

#     df = pd.read_csv(DATA_CSV)

#     # Clean numeric columns
#     df["dataset_count"] = pd.to_numeric(df.get("dataset_count"), errors="coerce")
#     df["total_tb"] = pd.to_numeric(df.get("total_tb"), errors="coerce")

#     # Helpful derived columns for nicer labeling/hover logic later
#     df["label"] = df["label"].astype(str)
#     return df


# # -----------------------------
# # Plots
# # -----------------------------
# @dataclass
# class PlotConfig:
#     top_n: int = 15
#     wrap_width: int = 28
#     figsize: tuple = (11, 7)


# def plot_largest_by_size(df: pd.DataFrame, cfg: PlotConfig = PlotConfig()) -> Path:
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     out = OUT_DIR / "largest_by_size_tb.png"

#     d = df.dropna(subset=["total_tb"]).sort_values("total_tb", ascending=False).head(cfg.top_n).copy()
#     if d.empty:
#         raise ValueError("No sources have total_tb populated.")

#     labels = _wrap_labels(d["label"].tolist(), width=cfg.wrap_width)
#     values = d["total_tb"].tolist()

#     fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=True)
#     ax.barh(labels, values)
#     ax.invert_yaxis()

#     ax.set_xlabel("Total size (TB)")
#     ax.set_title("Largest neuroscience data sources (by published / estimated size)")
#     ax.grid(True, axis="x", alpha=0.35)
#     ax.grid(False, axis="y")

#     _add_bar_labels(ax, values, fmt="auto")

#     # Subtitle / footnote-like context
#     ax.text(
#         0.0,
#         -0.10,
#         "Note: totals depend on each source’s published metric; some are estimates/fallbacks.",
#         transform=ax.transAxes,
#         fontsize=9,
#         alpha=0.85,
#         ha="left",
#         va="top",
#     )

#     fig.savefig(out)
#     plt.close(fig)
#     return out


# def plot_most_datasets(df: pd.DataFrame, cfg: PlotConfig = PlotConfig()) -> Path:
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     out = OUT_DIR / "most_datasets.png"

#     d = df.dropna(subset=["dataset_count"]).sort_values("dataset_count", ascending=False).head(cfg.top_n).copy()
#     if d.empty:
#         raise ValueError("No sources have dataset_count populated.")

#     labels = _wrap_labels(d["label"].tolist(), width=cfg.wrap_width)
#     values = d["dataset_count"].tolist()

#     fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=True)
#     ax.barh(labels, values)
#     ax.invert_yaxis()

#     ax.set_xlabel("Dataset / item count (units vary by source)")
#     ax.set_title("Neuroscience sources with the most datasets / items")
#     ax.grid(True, axis="x", alpha=0.35)
#     ax.grid(False, axis="y")

#     _add_bar_labels(ax, values, fmt="auto")

#     ax.text(
#         0.0,
#         -0.10,
#         "Tip: add a ‘unit’ column in metrics.csv (e.g., datasets, recordings, sessions) for clearer comparisons.",
#         transform=ax.transAxes,
#         fontsize=9,
#         alpha=0.85,
#         ha="left",
#         va="top",
#     )

#     fig.savefig(out)
#     plt.close(fig)
#     return out


# def plot_size_vs_count(df: pd.DataFrame, logx: bool = True, logy: bool = True) -> Path:
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     out = OUT_DIR / "size_vs_count.png"

#     d = df.dropna(subset=["total_tb", "dataset_count"]).copy()
#     fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

#     if d.empty:
#         ax.set_title("Size vs dataset count")
#         ax.text(0.5, 0.5, "No sources have BOTH total_tb and dataset_count.", ha="center", va="center")
#         ax.axis("off")
#         fig.savefig(out)
#         plt.close(fig)
#         return out

#     # Scatter
#     ax.scatter(d["dataset_count"], d["total_tb"], alpha=0.85)

#     # Log scales help readability if values span orders of magnitude
#     if logx:
#         ax.set_xscale("log")
#     if logy:
#         ax.set_yscale("log")

#     ax.set_xlabel("Dataset count" + (" (log scale)" if logx else ""))
#     ax.set_ylabel("Total size (TB)" + (" (log scale)" if logy else ""))
#     ax.set_title("Scale comparison: storage vs dataset count")

#     ax.grid(True, which="both", alpha=0.25)

#     # Simple label placement (kept dependency-free)
#     for _, row in d.iterrows():
#         x = row["dataset_count"]
#         y = row["total_tb"]
#         label = row["label"]
#         ax.annotate(
#             label,
#             (x, y),
#             textcoords="offset points",
#             xytext=(6, 5),
#             fontsize=9,
#             alpha=0.9,
#         )

#     fig.savefig(out)
#     plt.close(fig)
#     return out

# def plot_modality_distribution(df: pd.DataFrame) -> Path:
#     out = OUT_DIR / "modality_pie_chart.png"
    
#     # Example grouping (requires a 'modality' column in your CSV)
#     # Common categories: Electrophysiology, MRI, Microscopy, Behavioral
#     modality_counts = df.groupby('modality')['total_tb'].sum().sort_values(ascending=False)
    
#     fig, ax = plt.subplots(figsize=(10, 8))
#     colors = plt.cm.Pastel1(range(len(modality_counts)))
    
#     wedges, texts, autotexts = ax.pie(
#         modality_counts, 
#         labels=modality_counts.index, 
#         autopct='%1.1f%%',
#         startangle=140, 
#         colors=colors,
#         pctdistance=0.85,
#         explode=[0.05] * len(modality_counts) # Slight separation
#     )
    
#     # Add a center circle for a "Doughnut" look (cleaner than a standard pie)
#     centre_circle = plt.Circle((0,0), 0.70, fc='white')
#     fig.gca().add_artist(centre_circle)
    
#     ax.set_title("Global Neuroscience Data Volume by Modality (2026)", fontsize=16)
#     fig.savefig(out)
#     return out
# def main() -> int:
#     _apply_theme()
#     df = _read()

#     cfg = PlotConfig(top_n=15, wrap_width=30, figsize=(12, 7))

#     p1 = plot_largest_by_size(df, cfg)
#     p2 = plot_most_datasets(df, cfg)
#     p3 = plot_size_vs_count(df, logx=True, logy=True)

#     print("Wrote:")
#     print(f" - {p1}")
#     print(f" - {p2}")
#     print(f" - {p3}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
