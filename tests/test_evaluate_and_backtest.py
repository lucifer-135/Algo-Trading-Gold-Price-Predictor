import numpy as np
import pandas as pd
import pytest

from goldpred.backtest import backtest, buy_and_hold, drawdown_curve, performance
from goldpred.evaluate import walk_forward, walk_forward_splits
from goldpred.features import build_features
from goldpred.models import MODEL_NAMES, make_model
from goldpred.report import compare_models


def test_walk_forward_trains_only_on_the_past(prices):
    index = build_features(prices).index
    folds = list(walk_forward_splits(index, first_test_year=2016))
    assert [year for year, *_ in folds] == [2016]
    for _, train, test in folds:
        assert index[train].max() < index[test].min()
        assert not (train & test).any()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_produces_probabilities(prices, name):
    oos = walk_forward(build_features(prices), name, first_test_year=2016)
    assert oos["proba_up"].between(0, 1).all()
    assert (oos.index.year == 2016).all()


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        make_model("Crystal Ball")


def test_backtest_math():
    returns = pd.Series([0.01, -0.02, 0.03, 0.01])
    position = pd.Series([1, 0, 1, 1])
    # 3 position changes (enter, exit, re-enter) at 10 bps each.
    expected = [0.01 - 0.001, 0.0 - 0.001, 0.03 - 0.001, 0.01]
    np.testing.assert_allclose(backtest(returns, position, cost_bps=10), expected)
    # Buy & hold with no cost is just the market.
    np.testing.assert_allclose(buy_and_hold(returns), returns)


def test_performance_metrics():
    daily = pd.Series([0.10, -0.50, 0.20])  # equity: 1.10, 0.55, 0.66
    stats = performance(daily, position=pd.Series([1, 1, 0]))
    assert stats["total_return"] == pytest.approx(-0.34)
    assert stats["max_drawdown"] == pytest.approx(-0.50)
    assert stats["exposure"] == pytest.approx(2 / 3)
    assert stats["trades"] == 2
    # A loss on day one is a drawdown from the initial $1.
    assert drawdown_curve(pd.Series([-0.1, 0.05])).iloc[0] == pytest.approx(-0.1)


def test_majority_baseline_matches_buy_and_hold_in_rising_market(prices):
    data = build_features(prices)
    up = np.arange(len(data)) % 10 != 0  # 90% up days
    data["return"] = np.where(up, 1, -1) * data["return"].abs()
    data["target"] = up.astype(int)
    table = compare_models(data, first_test_year=2016, cost_bps=5,
                           model_names=["Majority class (baseline)"])
    baseline = table.loc["Majority class (baseline)"]
    market = table.loc["Buy & hold gold"]
    assert baseline["cagr"] == pytest.approx(market["cagr"])
    assert baseline["trades"] == market["trades"] == 1
