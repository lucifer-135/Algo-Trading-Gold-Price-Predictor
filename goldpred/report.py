"""Turn walk-forward predictions into trading results and a model comparison table."""

import pandas as pd

from goldpred.backtest import backtest, buy_and_hold, performance
from goldpred.evaluate import classification_metrics, walk_forward
from goldpred.models import MODEL_NAMES


def strategy_returns(oos: pd.DataFrame, threshold: float = 0.5, cost_bps: float = 0.0):
    """Go long gold when P(up) > threshold, otherwise hold cash. Returns (daily_returns, position)."""
    position = (oos["proba_up"] > threshold).astype(float)
    return backtest(oos["return"], position, cost_bps), position


def summarize(
    oos_by_model: dict[str, pd.DataFrame], threshold: float = 0.5, cost_bps: float = 5.0
) -> pd.DataFrame:
    """One row per model (plus buy & hold) with out-of-sample classification and trading metrics."""
    rows = {}
    for name, oos in oos_by_model.items():
        daily, position = strategy_returns(oos, threshold, cost_bps)
        rows[name] = {**classification_metrics(oos, threshold), **performance(daily, position)}
    market = buy_and_hold(oos["return"], cost_bps)
    rows["Buy & hold gold"] = performance(market, pd.Series(1.0, index=market.index))
    return pd.DataFrame(rows).T


def compare_models(
    data: pd.DataFrame,
    first_test_year: int = 2010,
    threshold: float = 0.5,
    cost_bps: float = 5.0,
    model_names: list[str] = MODEL_NAMES,
) -> pd.DataFrame:
    """Walk-forward every model on `data` and summarize the results."""
    oos = {name: walk_forward(data, name, first_test_year) for name in model_names}
    return summarize(oos, threshold, cost_bps)


def _fmt(fmt: str):
    return lambda x: "" if pd.isna(x) else format(x, fmt)


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version of `compare_models` output."""
    out = pd.DataFrame(index=table.index)
    out["Accuracy"] = [
        "" if pd.isna(acc) else f"{acc:.1%} ± {ci:.1%}"
        for acc, ci in zip(table["accuracy"], table["accuracy_ci95"])
    ]
    out["ROC AUC"] = table["roc_auc"].map(_fmt(".3f"))
    out["CAGR"] = table["cagr"].map(_fmt(".1%"))
    out["Sharpe"] = table["sharpe"].map(_fmt(".2f"))
    out["Max drawdown"] = table["max_drawdown"].map(_fmt(".1%"))
    out["Time in market"] = table["exposure"].map(_fmt(".0%"))
    out["Trades"] = table["trades"].map(_fmt(".0f"))
    return out


def to_markdown(table: pd.DataFrame) -> str:
    """Render a small string table as GitHub markdown (avoids a `tabulate` dependency)."""
    header = ["Model", *table.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join([str(i), *map(str, row)]) + " |" for i, row in table.iterrows()]
    return "\n".join(lines)
