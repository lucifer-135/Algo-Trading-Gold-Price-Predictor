"""Gold price direction forecasting: data loading, features, walk-forward evaluation and backtesting."""

from goldpred.backtest import backtest, buy_and_hold, performance
from goldpred.data import load_prices
from goldpred.evaluate import classification_metrics, walk_forward, walk_forward_splits
from goldpred.features import FEATURES, build_features
from goldpred.models import MODEL_NAMES, make_model

__all__ = [
    "FEATURES",
    "MODEL_NAMES",
    "backtest",
    "build_features",
    "buy_and_hold",
    "classification_metrics",
    "load_prices",
    "make_model",
    "performance",
    "walk_forward",
    "walk_forward_splits",
]
