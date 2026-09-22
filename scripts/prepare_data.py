"""Track A Step 1: load raw Plan-B demand, validate it, save the cleaned trace.

Usage:
    python scripts/prepare_data.py [--raw-path PATH] [--timestamp-col NAME] [--demand-col NAME]

If --raw-path is omitted, the script looks for a single .csv/.parquet file
under data/raw/<plan>/ (per configs/base.yaml). Raw Alibaba trace files are
never committed to this repo (see README "Data policy"), so this file must
be placed locally first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import DataQualityError, load_demand
from src.data.schema import SchemaError
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_raw_file(plan: str) -> Path:
    raw_dir = REPO_ROOT / "data" / "raw" / plan
    candidates = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet")))
    if not candidates:
        raise FileNotFoundError(
            f"No raw data file found in {raw_dir}. Place the Plan-B trace there "
            f"(see README 'Data policy'), or pass --raw-path explicitly."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple raw data files found in {raw_dir}: {candidates}. "
            f"Pass --raw-path to disambiguate."
        )
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--demand-col", default="demand")
    args = parser.parse_args()

    base_cfg = load_config("base.yaml")
    dt_seconds = base_cfg["data"]["dt_seconds"]
    plan = base_cfg["data"]["plan"]

    raw_path = args.raw_path or find_raw_file(plan)

    try:
        demand_df = load_demand(
            raw_path,
            timestamp_col=args.timestamp_col,
            demand_col=args.demand_col,
            dt_seconds=dt_seconds,
        )
    except (DataQualityError, SchemaError) as e:
        print(f"Data quality check failed: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = REPO_ROOT / "data" / "processed" / "demand"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demand.parquet"
    demand_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(demand_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
