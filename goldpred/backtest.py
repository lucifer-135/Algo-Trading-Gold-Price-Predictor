"""Long/flat trading backtest with transaction costs and standard performance metrics."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _turnover(position: pd.Series) -> pd.Series:
    """Notional traded each day; entering the first position counts as a trade."""
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    return turnover


def backtest(returns: pd.Series, position: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """Daily strategy returns, net of costs.

    `position[t]` is the exposure (0 = cash, 1 = long gold) held over day t, decided
    at the previous fix. Every change in position pays `cost_bps` on the traded notional.
    """
    position = position.astype(float)
    return position * returns - _turnover(position) * cost_bps / 1e4


def buy_and_hold(returns: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """Daily returns of simply holding gold for the whole period."""
    return backtest(returns, pd.Series(1.0, index=returns.index), cost_bps)


def equity_curve(daily: pd.Series) -> pd.Series:
    """Growth of $1 invested at the start of the series."""
    return (1 + daily).cumprod()


def drawdown_curve(daily: pd.Series) -> pd.Series:
    """Fraction below the running peak of the equity curve (0 at new highs).

    The peak starts at the initial $1, so an immediate loss counts as a drawdown.
    """
    equity = equity_curve(daily)
    return equity / equity.cummax().clip(lower=1.0) - 1


def performance(daily: pd.Series, position: pd.Series | None = None) -> dict:
    """Summary statistics for a series of daily returns (as fractions)."""
    final = equity_curve(daily).iloc[-1]
    std = daily.std()
    stats = {
        "total_return": final - 1,
        "cagr": final ** (TRADING_DAYS / len(daily)) - 1,
        "volatility": std * np.sqrt(TRADING_DAYS),
        "sharpe": daily.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "max_drawdown": drawdown_curve(daily).min(),
    }
    if position is not None:
        stats["exposure"] = position.mean()
        stats["trades"] = int(_turnover(position.astype(float)).sum())
    return stats
