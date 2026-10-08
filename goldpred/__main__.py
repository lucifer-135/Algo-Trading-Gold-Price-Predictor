"""Print the walk-forward model comparison: `python -m goldpred [--cost-bps 5] [--start 2010]`."""

import argparse
import sys

from goldpred.data import load_prices
from goldpred.features import build_features
from goldpred.report import compare_models, format_table, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2010, help="first out-of-sample year")
    parser.add_argument("--threshold", type=float, default=0.5, help="go long when P(up) > this")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit traded, in bps")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the table uses "±", which cp1252 consoles can't print

    data = build_features(load_prices())
    table = compare_models(data, args.start, args.threshold, args.cost_bps)
    oos_days = int(table["n_days"].dropna().iloc[0])
    print(
        f"Walk-forward test {args.start}-{data.index.year.max()} ({oos_days} trading days), "
        f"threshold {args.threshold}, costs {args.cost_bps:g} bps per trade\n"
    )
    print(to_markdown(format_table(table)))


if __name__ == "__main__":
    main()
