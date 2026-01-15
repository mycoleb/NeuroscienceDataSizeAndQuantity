
from typing import Optional



from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "metrics.csv"
OUT_DIR = ROOT / "outputs"




def _read() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing {DATA_CSV}. Run scripts/fetch_metrics.py first.")
    df = pd.read_csv(DATA_CSV)
    # Clean numeric columns
    df["dataset_count"] = pd.to_numeric(df["dataset_count"], errors="coerce")
    df["total_tb"] = pd.to_numeric(df["total_tb"], errors="coerce")
    return df


def plot_largest_by_size(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "largest_by_size_tb.png"

    d = df.dropna(subset=["total_tb"]).sort_values("total_tb", ascending=False).copy()
    if d.empty:
        raise ValueError("No sources have total_tb populated.")

    plt.figure()
    plt.barh(d["label"], d["total_tb"])
    plt.xlabel("Total size (TB)")
    plt.title("Largest neuroscience data sources (by published/estimated TB)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_most_datasets(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "most_datasets.png"

    d = df.dropna(subset=["dataset_count"]).sort_values("dataset_count", ascending=False).copy()
    if d.empty:
        raise ValueError("No sources have dataset_count populated.")

    plt.figure()
    plt.barh(d["label"], d["dataset_count"])
    plt.xlabel("Dataset count (unit depends on source)")
    plt.title("Neuroscience sources with the most datasets / items")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_size_vs_count(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "size_vs_count.png"

    d = df.dropna(subset=["total_tb", "dataset_count"]).copy()
    if d.empty:
        # Still generate a helpful chart shell
        plt.figure()
        plt.title("Size vs dataset count (need both fields populated)")
        plt.text(0.5, 0.5, "No sources have BOTH total_tb and dataset_count.", ha="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        return out

    plt.figure()
    plt.scatter(d["dataset_count"], d["total_tb"])
    for _, row in d.iterrows():
        plt.annotate(row["label"], (row["dataset_count"], row["total_tb"]), fontsize=8)

    plt.xlabel("Dataset count")
    plt.ylabel("Total size (TB)")
    plt.title("Scale comparison: storage vs dataset count")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main() -> int:
    df = _read()

    p1 = plot_largest_by_size(df)
    p2 = plot_most_datasets(df)
    p3 = plot_size_vs_count(df)

    print("Wrote:")
    print(f" - {p1}")
    print(f" - {p2}")
    print(f" - {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
