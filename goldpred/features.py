"""Feature engineering with a strict no-look-ahead rule.

Row t holds features built only from prices up to and including day t-1, and the
target is the direction of day t's return (PM fix t-1 -> PM fix t). A position
decided at day t-1's fix therefore earns exactly `return` on row t.

Note: the AM fix of day t is published *before* the PM fix of day t, but after the
moment the position must be taken, so using it as a feature would leak the answer.
"""

import pandas as pd

from goldpred.data import EUR_PRICE_COL, PRICE_COL

FEATURES = [
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "trend_short",
    "trend_long",
    "volatility",
    "rsi",
    "eurusd_ret_lag1",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (simple-moving-average variant), in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(
    prices: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    vol_window: int = 10,
) -> pd.DataFrame:
    """Return a frame with FEATURES, the day's `return` (as a fraction) and a binary `target`."""
    close = prices[PRICE_COL]
    ret = close.pct_change(fill_method=None)
    # Gold priced in USD and EUR implies the EUR/USD rate; a weaker dollar tends to lift gold.
    eurusd_ret = (close / prices[EUR_PRICE_COL]).pct_change(fill_method=None)
    past_close = close.shift(1)

    out = pd.DataFrame(index=prices.index)
    for lag in (1, 2, 3, 5):
        out[f"ret_lag{lag}"] = ret.shift(lag)
    out["trend_short"] = past_close / past_close.rolling(short_window).mean() - 1
    out["trend_long"] = past_close / past_close.rolling(long_window).mean() - 1
    out["volatility"] = ret.shift(1).rolling(vol_window).std()
    out["rsi"] = _rsi(past_close)
    out["eurusd_ret_lag1"] = eurusd_ret.shift(1)

    out["return"] = ret
    out["target"] = (ret > 0).astype(int)
    return out.dropna()
