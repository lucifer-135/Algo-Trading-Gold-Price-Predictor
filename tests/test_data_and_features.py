import numpy as np
import pandas as pd
import pytest

from goldpred.data import DATA_PATH, load_prices
from goldpred.features import FEATURES, build_features


def test_missing_pm_fix_days_are_dropped_without_losing_returns(tmp_path):
    csv = tmp_path / "prices.csv"
    csv.write_text(
        "Date,USD (AM),USD (PM),EURO (AM),EURO (PM)\n"
        "2018-12-21,100,100,90,90\n"
        "2018-12-24,101,,91,\n"  # Christmas Eve: AM fix only
        "2018-12-27,102,110,92,99\n"
    )
    prices = load_prices(csv)
    assert list(prices.index.strftime("%m-%d")) == ["12-21", "12-27"]
    # The return on the 27th is measured from the last real PM fix (the 21st), not lost.
    assert prices["USD (PM)"].pct_change().iloc[-1] == pytest.approx(0.10)


def test_real_dataset_loads_cleanly():
    prices = load_prices(DATA_PATH)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique
    assert prices[["USD (PM)", "EURO (PM)"]].notna().all().all()


def test_target_is_direction_of_same_day_return(prices):
    data = build_features(prices)
    assert set(FEATURES) <= set(data.columns)
    assert not data.isna().any().any()
    assert (data["target"] == (data["return"] > 0)).all()


def test_features_do_not_look_ahead(prices):
    """Changing the price on day t (and after) must not change any feature on day t."""
    base = build_features(prices)
    day = base.index[100]
    shocked = prices.copy()
    shocked.loc[day:, ["USD (PM)", "EURO (PM)"]] *= 1.5
    after = build_features(shocked)

    pd.testing.assert_frame_equal(base.loc[:day, FEATURES], after.loc[:day, FEATURES])
    # ...while the target on day t does see the shock, so the test would catch a leak.
    assert after.loc[day, "return"] != pytest.approx(base.loc[day, "return"])


def test_trend_feature_uses_previous_close(prices):
    data = build_features(prices, short_window=3)
    close = prices["USD (PM)"]
    day = data.index[50]
    i = close.index.get_loc(day)
    expected = close.iloc[i - 1] / close.iloc[i - 3 : i].mean() - 1
    assert data.loc[day, "trend_short"] == pytest.approx(expected)
    assert np.isfinite(data["rsi"]).all() and data["rsi"].between(0, 100).all()
