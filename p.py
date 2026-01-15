import csv
import datetime as dt
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "sources.yaml"
DATA_DIR = ROOT / "data"
OUT_CSV = DATA_DIR / "metrics.csv"


@dataclass
class MetricRow:
    timestamp_utc: str
    source_id: str
    label: str
    dataset_count: Optional[int]
    total_gb: Optional[float]
    total_tb: Optional[float]
    notes: str
    provenance_url: str
    citation: str
    fetch_status: str  # ok / fallback / error


def tb_to_gb(tb: float) -> float:
    # Use decimal TB->GB for “at scale” comms (1 TB = 1000 GB)
    return tb * 1000.0


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_sources():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow either:
    # 1) {"sources": [...]}   (preferred)
    # 2) [...]               (top-level list)
    if isinstance(cfg, dict) and "sources" in cfg:
        return cfg["sources"]
    if isinstance(cfg, list):
        return cfg

    raise ValueError("sources.yaml must be either a list of sources or a dict with a 'sources:' key.")



def fetch_dandi_totals(api_base: str, timeout_s: int = 30) -> Tuple[Optional[int], Optional[float]]:
    """
    Attempts to compute:
      - dataset_count (dandisets)
      - total_size_tb by summing the 'size' field if present.
    DANDI API is public; no auth required for read-only listing.
    """
    # Endpoint pattern from DANDI swagger/docs: GET /dandisets/
    # We page through results. If 'size' isn't present, we still return count.
    page_size = 100
    url = f"{api_base.rstrip('/')}/dandisets/?page_size={page_size}"
    total_count: Optional[int] = None
    total_bytes: int = 0
    saw_size_field = False

    while url:
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        j = r.json()

        # DANDI commonly provides a "count" and "results" list.
        if total_count is None:
            total_count = safe_int(j.get("count"))

        results = j.get("results") or []
        for item in results:
            # Different deployments may name this differently.
            # Try a few common keys.
            size = item.get("size") or item.get("total_size") or item.get("bytes")
            if size is not None:
                saw_size_field = True
                try:
                    total_bytes += int(size)
                except Exception:
                    pass

        url = j.get("next")

    if total_count is None:
        total_count = 0

    if not saw_size_field:
        # We can’t compute total TB reliably from listing if no size field is present.
        return total_count, None

    total_tb = (total_bytes / 1e12)  # bytes -> TB (decimal)
    return total_count, total_tb


def build_rows() -> List[MetricRow]:
    sources = load_sources()
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    rows: List[MetricRow] = []

    for src in sources:
        kind = src["kind"]
        source_id = src["id"]
        label = src["label"]

        dataset_count = None
        total_tb = None
        notes = ""
        prov = ""
        citation = ""
        status = "ok"

        try:
            if kind == "static":
                dataset_count = safe_int(src.get("dataset_count"))
                total_tb = safe_float(src.get("total_tb"))
                notes = src.get("notes", "")
                prov = src.get("provenance_url", "")
                citation = src.get("citation", "")

            elif kind == "dandi_api":
                api_base = src["api_base"]
                prov = src.get("fallback", {}).get("provenance_url", api_base)
                citation = src.get("fallback", {}).get("citation", "DANDI API + fallback")

                api_count, api_tb = fetch_dandi_totals(api_base=api_base)
                if api_count is not None:
                    dataset_count = api_count
                if api_tb is not None and api_tb > 0:
                    total_tb = api_tb
                    notes = "Computed from DANDI API listing."
                    status = "ok"
                else:
                    fb = src.get("fallback", {})
                    dataset_count = safe_int(fb.get("dataset_count"))
                    total_tb = safe_float(fb.get("total_tb"))
                    notes = fb.get("notes", "Fallback used (API did not expose total size).")
                    prov = fb.get("provenance_url", prov)
                    citation = fb.get("citation", citation)
                    status = "fallback"
            else:
                raise ValueError(f"Unknown kind: {kind}")

        except Exception as e:
            # Hard fallback to static fields if present
            fb = src.get("fallback") or {}
            dataset_count = safe_int(fb.get("dataset_count")) or safe_int(src.get("dataset_count"))
            total_tb = safe_float(fb.get("total_tb")) or safe_float(src.get("total_tb"))
            notes = f"Error fetching live metrics: {e.__class__.__name__}: {e}"
            prov = fb.get("provenance_url", src.get("provenance_url", ""))
            citation = fb.get("citation", src.get("citation", ""))
            status = "error"

        total_gb = tb_to_gb(total_tb) if total_tb is not None else None

        rows.append(
            MetricRow(
                timestamp_utc=now,
                source_id=source_id,
                label=label,
                dataset_count=dataset_count,
                total_gb=total_gb,
                total_tb=total_tb,
                notes=notes,
                provenance_url=prov,
                citation=citation,
                fetch_status=status,
            )
        )

    return rows


def write_csv(rows: List[MetricRow]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp_utc",
                "source_id",
                "label",
                "dataset_count",
                "total_gb",
                "total_tb",
                "notes",
                "provenance_url",
                "citation",
                "fetch_status",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.timestamp_utc,
                    r.source_id,
                    r.label,
                    r.dataset_count if r.dataset_count is not None else "",
                    f"{r.total_gb:.3f}" if r.total_gb is not None else "",
                    f"{r.total_tb:.6f}" if r.total_tb is not None else "",
                    r.notes,
                    r.provenance_url,
                    r.citation,
                    r.fetch_status,
                ]
            )


def main() -> int:
    rows = build_rows()
    write_csv(rows)
    print(f"Wrote: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
