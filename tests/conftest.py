import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    """Two years of synthetic random-walk gold prices in the same layout as the real CSV."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2016-12-31", name="Date")
    usd = 1200 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    eurusd = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates))))
    return pd.DataFrame(
        {"USD (AM)": usd, "USD (PM)": usd, "EURO (AM)": usd / eurusd, "EURO (PM)": usd / eurusd},
        index=dates,
    )
