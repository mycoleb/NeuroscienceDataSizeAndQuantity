# Neuroscience Data Sources at Scale

This project visualizes **(1) the largest neuroscience data sources by storage size** and **(2) which sources have the most datasets/items**.

It produces:
- `outputs/largest_by_size_tb.png`
- `outputs/most_datasets.png`
- `outputs/size_vs_count.png`

## What counts as “dataset” here?

Different repositories use different units:
- DANDI: *dandisets*
- OpenNeuro: *public datasets*
- NeuroMorpho: *digital reconstructions* (not “datasets” in the same sense)
- Allen BICCN: headline storage for a major imaging corpus (dataset count not represented as one simple number)

So the “most datasets” chart is best read as “largest collections by their native unit”.

## Data sources (high level)

Values are either:
1) fetched from a public API when feasible (DANDI), or  
2) taken from public published headline metrics (OpenNeuro count, Allen TB, NeuroMorpho reconstructions, HCP S500 size).

The provenance URL + citation text are stored in `data/metrics.csv`.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

python scripts/fetch_metrics.py
python scripts/plot_metrics.py
