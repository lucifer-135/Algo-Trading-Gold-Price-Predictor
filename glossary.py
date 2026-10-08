"""Plain-English definitions shown in the dashboard's tooltips and Glossary tab.

Keeping them in one place means a hover tooltip and the glossary never disagree.
"""

TERMS = {
    "Accuracy": (
        "Share of test days on which the model called the direction (up or down) correctly. "
        "On its own it means little: compare it with the baseline."
    ),
    "Baseline (majority class)": (
        "A deliberately naive model that always predicts whichever direction was more common in its "
        "training years (in this dataset that's always \"up\"). Any real model must beat it to be useful."
    ),
    "Basis point (bp)": "One hundredth of a percent: 1 bp = 0.01%, so 5 bps = 0.05%.",
    "Buy & hold": (
        "The simplest strategy: buy gold on the first test day and never sell. "
        "It's the benchmark the trading strategy has to beat."
    ),
    "CAGR": (
        "Compound Annual Growth Rate: the steady yearly return that would turn the starting amount "
        "into the final amount. For example, 3.6% a year over ten years is about +42% in total."
    ),
    "Drawdown": (
        "How far an investment has fallen below its previous high point. "
        "**Max drawdown** is the worst such fall over the whole period; closer to 0% is better."
    ),
    "EUR/USD feature": (
        "Gold's USD price divided by its EUR price gives the euro-dollar exchange rate. "
        "Yesterday's change in it measures dollar strength; a weaker dollar tends to push gold up."
    ),
    "Feature": "An input the model learns from, such as yesterday's return or recent volatility.",
    "Lagged return": "A past daily return used as a feature. *Lag 1* is yesterday's, *lag 5* is from 5 trading days ago.",
    "LBMA PM fix": (
        "The official gold price set each afternoon in London by the London Bullion Market Association. "
        "This app predicts whether today's PM fix is above yesterday's."
    ),
    "Long / cash": (
        "The strategy either holds gold for the day (*long*) or sits in cash earning 0%. "
        "It never bets on prices falling (no short selling)."
    ),
    "Look-ahead bias": (
        "Accidentally letting a model see information that wasn't available yet when the decision had "
        "to be made. It makes backtests look far better than reality. Here every feature for a day "
        "uses data up to the *previous* day only, and a unit test checks this."
    ),
    "Noise band (95% CI)": (
        "The range within which accuracy could land purely by luck. If a model beats the baseline by "
        "less than this, the difference isn't evidence of skill."
    ),
    "Out-of-sample": "Data the model never saw during training. Only out-of-sample results show real predictive power.",
    "Overfitting": (
        "Tuning a model (or its settings) until it fits past data, including its random noise, so well "
        "that it fails on new data. Moving these sliders until the backtest looks great does this too."
    ),
    "P(up) threshold": (
        "Each model outputs a probability that gold will rise. The strategy holds gold only when that "
        "probability is above the threshold; a higher threshold means fewer, more confident trades."
    ),
    "Return": "The percentage change in price from one day's PM fix to the next.",
    "ROC AUC": (
        "How well the model ranks up days above down days, regardless of the threshold. "
        "0.5 is a coin flip and 1.0 is perfect; values near 0.51 mean almost no signal."
    ),
    "RSI": (
        "Relative Strength Index (14 days): a 0–100 momentum gauge comparing recent gains to losses. "
        "Above 70 is often called \"overbought\" and below 30 \"oversold\"."
    ),
    "Sharpe ratio": (
        "Return earned per unit of risk: average yearly return divided by yearly volatility. "
        "Higher is better. It lets you compare strategies that take different amounts of risk."
    ),
    "Time in market": "Share of days the strategy held gold rather than cash.",
    "Trade": "Any change of position (buying into gold or selling out to cash). Each one pays the transaction cost.",
    "Transaction cost": (
        "What it costs to buy or sell: broker fees plus the bid–ask spread. It is charged on every trade, "
        "so strategies that trade often are hit hardest."
    ),
    "Trend features": (
        "Yesterday's price compared with its average over the last few days (short) and weeks (long). "
        "Positive means the price is above its recent average, i.e. trending up."
    ),
    "Volatility": "How much daily returns swing around, measured as their standard deviation over recent days.",
    "Walk-forward validation": (
        "Testing a model the way it would actually be used: train on all years before year *Y*, predict "
        "*Y*, then move forward one year and repeat. The model never trains on the future."
    ),
}

MODEL_INFO = {
    "Majority class (baseline)": "Always predicts the most common direction from training. The benchmark to beat.",
    "Logistic Regression": "A linear model that weighs each feature to estimate the chance of an up day. Simple and hard to overfit.",
    "Random Forest": "Averages 300 decision trees, each built on a random sample of days. Can capture non-linear patterns.",
    "Gradient Boosting": "Builds small decision trees one after another, each fixing the previous ones' mistakes.",
}

COLUMN_HELP = {
    "Accuracy": TERMS["Accuracy"] + " ± shows the 95% noise band.",
    "ROC AUC": TERMS["ROC AUC"],
    "CAGR": TERMS["CAGR"] + " After transaction costs.",
    "Sharpe": TERMS["Sharpe ratio"],
    "Max drawdown": TERMS["Drawdown"],
    "Time in market": TERMS["Time in market"],
    "Trades": TERMS["Trade"],
}
