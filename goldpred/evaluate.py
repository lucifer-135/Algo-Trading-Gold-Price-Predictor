"""Expanding-window walk-forward evaluation.

For every test year Y the model is trained on all data before Y and then predicts
Y, the way it would have been used in real time. The concatenated out-of-sample
predictions cover many market regimes instead of a single, possibly lucky, test year.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from goldpred.features import FEATURES
from goldpred.models import make_model


def walk_forward_splits(
    index: pd.DatetimeIndex, first_test_year: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (test_year, train_mask, test_mask) for each year from `first_test_year` on."""
    years = index.year
    for year in sorted(set(years[years >= first_test_year])):
        yield year, years < year, years == year


def walk_forward(data: pd.DataFrame, model_name: str, first_test_year: int = 2010) -> pd.DataFrame:
    """Return out-of-sample P(up) for every test day, with its true target and return."""
    folds = []
    for year, train, test in walk_forward_splits(data.index, first_test_year):
        model = make_model(model_name).fit(data.loc[train, FEATURES], data.loc[train, "target"])
        folds.append(
            pd.DataFrame(
                {
                    "proba_up": model.predict_proba(data.loc[test, FEATURES])[:, 1],
                    "target": data.loc[test, "target"],
                    "return": data.loc[test, "return"],
                    "fold": year,
                },
                index=data.index[test],
            )
        )
    if not folds:
        raise ValueError(f"No data on or after {first_test_year} to test on")
    return pd.concat(folds)


def classification_metrics(oos: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Accuracy-style metrics for out-of-sample predictions from `walk_forward`."""
    pred = (oos["proba_up"] > threshold).astype(int)
    y = oos["target"]
    n = len(y)
    acc = accuracy_score(y, pred)
    return {
        "accuracy": acc,
        # 95% normal-approximation interval: is the edge distinguishable from luck?
        "accuracy_ci95": 1.96 * np.sqrt(acc * (1 - acc) / n),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, oos["proba_up"]) if y.nunique() == 2 else np.nan,
        "up_day_rate": y.mean(),
        "predicted_up_rate": pred.mean(),
        "n_days": n,
    }
