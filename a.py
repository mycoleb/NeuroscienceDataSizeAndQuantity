from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents
IN_CSV = ROOT / "data" / "metrics.csv"
OUT_DIR = ROOT / "outputs"


# ----------------------------
# Helpers
# ----------------------------
def _norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _coerce_num(series: pd.Series) -> pd.Series:
    """
    Convert values like "1,234", " 56 ", "N/A" into numeric.
    """
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": None, "none": None, "nan": None, "n/a": None, "na": None})
    return pd.to_numeric(s, errors="coerce")


def _parse_size_to_tb(row: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    """
    Try multiple schemas:
    1) total_tb already present
    2) total_gb present
    3) total_size + size_unit (e.g., 1200 + 'GB')
    4) size (string) like '1.2 TB', '800GB', '0.5PB'

    Returns (total_tb, reason)
    """
    # 1
    if "total_tb" in row and pd.notna(row["total_tb"]):
        return float(row["total_tb"]), "from total_tb"

    # 2
    if "total_gb" in row and pd.notna(row["total_gb"]):
        return float(row["total_gb"]) / 1024.0, "from total_gb"

    # 3
    if "total_size" in row and pd.notna(row["total_size"]):
        val = row["total_size"]
        unit = str(row.get("size_unit", "")).strip().lower()
        if isinstance(val, str):
            try:
                val = float(val.replace(",", "").strip())
            except Exception:
                val = math.nan
        if pd.notna(val) and unit:
            factor = {
                "b": 1 / (1024**4),
                "kb": 1 / (1024**3),
                "mb": 1 / (1024**2),
                "gb": 1 / 1024,
                "tb": 1.0,
                "pb": 1024.0,
            }.get(unit)
            if factor is not None:
                return float(val) * factor, f"from total_size + size_unit({unit})"

    # 4
    raw = row.get("size") or row.get("total_size_text") or row.get("data_size")
    if raw is not None and pd.notna(raw):
        s = str(raw).strip().lower().replace(",", "")
        m = re.search(r"([0-9]*\.?[0-9]+)\s*(pb|tb|gb|mb|kb|b)\b", s)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            factor = {
                "b": 1 / (1024**4),
                "kb": 1 / (1024**3),
                "mb": 1 / (1024**2),
                "gb": 1 / 1024,
                "tb": 1.0,
                "pb": 1024.0,
            }[unit]
            return val * factor, f"from text size({unit})"

    return None, None


def _wrap_label(label: str, width: int = 30) -> str:
    label = str(label).strip()
    return "\n".join(textwrap.wrap(label, width=width)) if label else ""


# ----------------------------
# Main extraction
# ----------------------------
def extract_metrics(df: pd.DataFrame, wrap_width: int = 30) -> pd.DataFrame:
    # Normalize column names
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]

    # Canonical columns we like for viz (keep if present)
    # You can add more here any time without breaking old CSVs.
    wanted_cols = [
        "label",
        "source",
        "url",
        "category",
        "modality",
        "species",
        "license",
        "access",
        "last_updated",
        # sizes
        "total_tb",
        "total_gb",
        "total_size",
        "size_unit",
        "size",
        "total_size_text",
        "data_size",
        # counts
        "dataset_count",
        "count",
        "n_datasets",
        "n_records",
        "unit",
        "count_unit",
    ]
    existing = [c for c in wanted_cols if c in df.columns]
    if existing:
        df = df[existing].copy()

    # Ensure label exists (fallbacks)
    if "label" not in df.columns:
        # try common alternates
        for alt in ("name", "database", "dataset", "source"):
            if alt in df.columns:
                df["label"] = df[alt]
                break
        else:
            df["label"] = ""

    # Coerce counts: dataset_count is canonical
    if "dataset_count" not in df.columns:
        df["dataset_count"] = pd.NA

    # If dataset_count missing but other count columns exist, use first non-null
    for alt in ("n_datasets", "count", "n_records"):
        if alt in df.columns:
            mask = df["dataset_count"].isna()
            df.loc[mask, "dataset_count"] = df.loc[mask, alt]
            break

    df["dataset_count"] = _coerce_num(df["dataset_count"])

    # Coerce size fields and compute standardized total_tb
    for c in ("total_tb", "total_gb", "total_size"):
        if c in df.columns:
            df[c] = _coerce_num(df[c])

    tb_vals = []
    tb_src = []
    for _, row in df.iterrows():
        tb, why = _parse_size_to_tb(row)
        tb_vals.append(tb if tb is not None else math.nan)
        tb_src.append(why if why is not None else "")
    df["total_tb_std"] = pd.to_numeric(tb_vals, errors="coerce")
    df["total_tb_source"] = tb_src

    df["size_gb_std"] = df["total_tb_std"] * 1024.0

    # Flags
    df["has_size"] = df["total_tb_std"].notna()
    df["has_count"] = df["dataset_count"].notna()

    # Display label for plots
    df["display_label"] = df["label"].apply(lambda x: _wrap_label(x, width=wrap_width))

    # A single "unit" column (helpful if you later annotate charts)
    if "unit" not in df.columns:
        df["unit"] = pd.NA
    if "count_unit" in df.columns:
        df["unit"] = df["unit"].fillna(df["count_unit"])

    # Trim whitespace for common text cols
    for c in ("label", "source", "url", "category", "modality", "species", "license", "access", "unit"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": ""})

    return df


def build_summary(df: pd.DataFrame) -> Dict:
    def _n(x):  # json-safe numbers
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    summary = {
        "rows_total": int(len(df)),
        "with_size": int(df["has_size"].sum()) if "has_size" in df.columns else 0,
        "with_count": int(df["has_count"].sum()) if "has_count" in df.columns else 0,
        "missing_columns": [],
        "size_tb": {
            "min": _n(float(df["total_tb_std"].min())) if df["total_tb_std"].notna().any() else None,
            "median": _n(float(df["total_tb_std"].median())) if df["total_tb_std"].notna().any() else None,
            "max": _n(float(df["total_tb_std"].max())) if df["total_tb_std"].notna().any() else None,
        },
        "dataset_count": {
            "min": _n(float(df["dataset_count"].min())) if df["dataset_count"].notna().any() else None,
            "median": _n(float(df["dataset_count"].median())) if df["dataset_count"].notna().any() else None,
            "max": _n(float(df["dataset_count"].max())) if df["dataset_count"].notna().any() else None,
        },
        "missingness_by_column": {},
    }

    for c in df.columns:
        summary["missingness_by_column"][c] = float(df[c].isna().mean())

    return summary


def write_top_tables(df: pd.DataFrame, top_n: int) -> str:
    lines = []
    lines.append("# Metrics Top Tables\n")

    if df["total_tb_std"].notna().any():
        top_size = df.dropna(subset=["total_tb_std"]).sort_values("total_tb_std", ascending=False).head(top_n)
        lines.append(f"## Top {top_n} by size (TB)\n")
        lines.append(top_size[["label", "total_tb_std", "total_tb_source", "url"]].to_markdown(index=False))
        lines.append("\n")

    if df["dataset_count"].notna().any():
        top_count = df.dropna(subset=["dataset_count"]).sort_values("dataset_count", ascending=False).head(top_n)
        lines.append(f"## Top {top_n} by dataset_count\n")
        cols = ["label", "dataset_count", "unit", "url"]
        cols = [c for c in cols if c in top_count.columns]
        lines.append(top_count[cols].to_markdown(index=False))
        lines.append("\n")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default=str(IN_CSV), help="Input CSV (default: data/metrics.csv)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory (default: outputs/)")
    ap.add_argument("--wrap-width", type=int, default=30, help="Wrap width for display_label")
    ap.add_argument("--top-n", type=int, default=15, help="Top N rows for tables")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df_raw = pd.read_csv(in_csv)
    df = extract_metrics(df_raw, wrap_width=args.wrap_width)

    # Outputs
    clean_csv = out_dir / "metrics_clean.csv"
    summary_json = out_dir / "metrics_summary.json"
    tables_md = out_dir / "metrics_top_tables.md"

    df.to_csv(clean_csv, index=False)

    summary = build_summary(df)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tables_md.write_text(write_top_tables(df, top_n=args.top_n), encoding="utf-8")

    print("Wrote:")
    print(f" - {clean_csv}")
    print(f" - {summary_json}")
    print(f" - {tables_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())