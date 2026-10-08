"""Load and clean the daily LBMA gold price fixes."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_price.csv"
PRICE_COL = "USD (PM)"
EUR_PRICE_COL = "EURO (PM)"


def load_prices(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Return the price table indexed by date, sorted, with no-PM-fix days removed.

    The London PM fix is not held on the last trading day before Christmas or before
    New Year (usually 24 and 31 December), so those rows only have AM prices. They are dropped *before* returns are computed: the next
    day's return is then measured from the last real PM fix, so no return is lost
    and results don't depend on how a pandas version fills NaNs in pct_change.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.dropna(subset=[PRICE_COL, EUR_PRICE_COL])
