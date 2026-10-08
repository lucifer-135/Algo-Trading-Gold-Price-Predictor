"""Streamlit dashboard: walk-forward gold direction forecasting with a cost-aware backtest.

Run with `streamlit run app.py`.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from glossary import COLUMN_HELP, MODEL_INFO, TERMS
from goldpred.backtest import buy_and_hold, drawdown_curve, equity_curve, performance
from goldpred.data import load_prices
from goldpred.evaluate import classification_metrics, walk_forward
from goldpred.features import build_features
from goldpred.models import MODEL_NAMES
from goldpred.report import format_table, strategy_returns, summarize

GOLD, GREY = "#D4A017", "#8C8C8C"
FILL = {GOLD: "rgba(212, 160, 23, 0.25)", GREY: "rgba(140, 140, 140, 0.25)"}
BASELINE = "Majority class (baseline)"

st.set_page_config(page_title="Gold Direction Forecasting", page_icon="🪙", layout="wide")


def pp(diff: float) -> str:
    """Format a difference between two percentages as percentage points."""
    return f"{diff * 100:+.1f} pp"


# ---------- cached computation ----------
@st.cache_data
def get_features(short_window: int, long_window: int, vol_window: int) -> pd.DataFrame:
    return build_features(load_prices(), short_window, long_window, vol_window)


@st.cache_data(show_spinner="Training walk-forward models…")
def get_predictions(model_name: str, first_test_year: int, windows: tuple[int, int, int]) -> pd.DataFrame:
    return walk_forward(get_features(*windows), model_name, first_test_year)


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Hover the small **?** icons for definitions, or open the 📖 Glossary tab.")
    model_name = st.selectbox(
        "Model", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest"),
        help="The classifier being tested. Pick the baseline to see the benchmark every model has to beat.",
    )
    st.caption(MODEL_INFO[model_name])
    first_test_year = st.slider(
        "First out-of-sample year", 2006, 2018, 2010,
        help="Where testing starts. Each year from here on is predicted by a model trained only on "
        "earlier years (walk-forward validation). Earlier start = more test years, less training data.",
    )
    st.subheader("Strategy")
    threshold = st.slider(
        "Go long when P(up) >", 0.40, 0.60, 0.50, 0.01,
        help=TERMS["P(up) threshold"] + " Otherwise the strategy holds cash for the day.",
    )
    cost_bps = st.slider(
        "Transaction cost (bps per trade)", 0.0, 20.0, 5.0, 0.5,
        help=TERMS["Transaction cost"] + " " + TERMS["Basis point (bp)"],
    )
    with st.expander("Feature windows (trading days)"):
        st.caption("How many past trading days each feature looks back over.")
        short_window = st.slider(
            "Short trend", 3, 20, 5,
            help="Length of the short moving average. The feature is yesterday's price relative to it.",
        )
        long_window = st.slider(
            "Long trend", 10, 100, 20,
            help="Length of the long moving average. Captures slower, multi-week trends.",
        )
        vol_window = st.slider("Volatility", 5, 60, 10, help=TERMS["Volatility"])
    st.caption(
        "Defaults were fixed before looking at test results. Sliding these until the "
        "backtest looks good is overfitting to the test period, not a better model."
    )

windows = (short_window, long_window, vol_window)
oos = get_predictions(model_name, first_test_year, windows)
baseline_oos = get_predictions(BASELINE, first_test_year, windows)

clf = classification_metrics(oos, threshold)
base_clf = classification_metrics(baseline_oos)  # the baseline always uses its majority class
strat_daily, position = strategy_returns(oos, threshold, cost_bps)
market_daily = buy_and_hold(oos["return"], cost_bps)
strat = performance(strat_daily, position)
market = performance(market_daily)
test_period = f"{oos.index.min():%b %Y} – {oos.index.max():%b %Y}"

# ---------- header ----------
st.title("🪙 Gold Price Direction Forecasting")
st.markdown(
    f"Can machine learning predict whether gold's **London PM fix** will close up or down "
    f"tomorrow? **{model_name}** is evaluated walk-forward on **{clf['n_days']:,} unseen "
    f"trading days** ({test_period}) and traded as a long/cash strategy after costs."
)

with st.expander("🧭 New here? How to use this dashboard"):
    st.markdown(
        """
1. **Pick a model** in the sidebar (on mobile, tap **›** at the top left to open it). Each model
   is trained only on years *before* the one it predicts, exactly as it would be used in real life.
2. **Read the five numbers below.** The small coloured tag under each compares the model with the
   naive *baseline* or with simply *buying and holding gold*: green is better, red is worse.
3. **Read the verdict box.** It sums up whether the model shows real skill and whether that
   skill survives trading costs.
4. **Explore the tabs:**
   - 📈 **Backtest**: how \\$1 would have grown trading on the model, and how costs erode it.
   - 🧪 **Model comparison**: all four models side by side on identical test data.
   - 📅 **Year by year**: whether results are consistent or driven by a few lucky years.
   - 🔍 **Methodology**: how the data, features and testing work.
   - 📖 **Glossary**: plain-English definitions of every term, with search.
5. **Hover the small ? icons** (or a table column header) for a quick definition.

⚠️ *Changing the sliders until the backtest looks great is overfitting: you're fitting to the
test period, not finding a better model. The defaults were fixed before any results were seen.*
"""
    )

cols = st.columns(5)
cols[0].metric(
    "Accuracy", f"{clf['accuracy']:.1%}",
    f"{pp(clf['accuracy'] - base_clf['accuracy'])} vs baseline",
    help=f"{TERMS['Accuracy']}\n\nThe noise band for {clf['n_days']:,} days is ±{clf['accuracy_ci95']:.1%}: "
    f"a smaller gap to the baseline ({base_clf['accuracy']:.1%}) could be pure luck.",
)
cols[1].metric("ROC AUC", f"{clf['roc_auc']:.3f}", help=TERMS["ROC AUC"])
cols[2].metric(
    "CAGR", f"{strat['cagr']:.1%}", f"{pp(strat['cagr'] - market['cagr'])} vs buy & hold",
    help=f"{TERMS['CAGR']}\n\nStrategy after costs vs. buy & hold ({market['cagr']:.1%}).",
)
cols[3].metric(
    "Sharpe ratio", f"{strat['sharpe']:.2f}", f"{strat['sharpe'] - market['sharpe']:+.2f} vs buy & hold",
    help=f"{TERMS['Sharpe ratio']}\n\nBuy & hold gold scored {market['sharpe']:.2f}.",
)
cols[4].metric(
    "Max drawdown", f"{strat['max_drawdown']:.1%}",
    f"{pp(strat['max_drawdown'] - market['max_drawdown'])} vs buy & hold",
    help=f"{TERMS['Drawdown']}\n\nBuy & hold's worst fall was {market['max_drawdown']:.1%}; "
    "a green tag means the strategy's worst fall was smaller.",
)

edge = clf["accuracy"] - base_clf["accuracy"]
significant = edge > clf["accuracy_ci95"]
beats_market = strat["sharpe"] > market["sharpe"]
verdict = (
    f"Accuracy is {pp(edge)} versus the naive baseline, "
    + ("**outside**" if significant else "**inside**")
    + f" the ±{clf['accuracy_ci95'] * 100:.1f} pp band you'd expect from luck alone. "
    f"After {cost_bps:g} bps costs the strategy's risk-adjusted return is "
    + ("**better**" if beats_market else "**worse**")
    + f" than simply holding gold (Sharpe {strat['sharpe']:.2f} vs {market['sharpe']:.2f})."
)
(st.success if significant and beats_market else st.info)(verdict, icon="🧭")

tab_backtest, tab_models, tab_years, tab_method, tab_glossary = st.tabs(
    ["📈 Backtest", "🧪 Model comparison", "📅 Year by year", "🔍 Methodology", "📖 Glossary"]
)

# ---------- backtest ----------
with tab_backtest:
    st.caption(
        "**How to read this:** the top chart shows how **\\$1** invested on the first test day would have "
        "grown, for the model's strategy (gold) and for buying and holding gold (grey). The bottom chart "
        "shows the **drawdown**: how far each was below its previous peak. Shallower is better. "
        "Hover the chart to see the values on any date."
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    for daily, name, color in [(market_daily, "Buy & hold gold", GREY), (strat_daily, f"{model_name} strategy", GOLD)]:
        fig.add_scatter(x=daily.index, y=equity_curve(daily), name=name, line=dict(color=color), row=1, col=1)
        fig.add_scatter(
            x=daily.index, y=drawdown_curve(daily), name=name, line=dict(color=color, width=1),
            fill="tozeroy", fillcolor=FILL[color], showlegend=False, row=2, col=1,
        )
    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.06, x=0))
    st.plotly_chart(fig, width="stretch")

    years_tested = len(oos) / 252
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time in market", f"{strat['exposure']:.0%}", help=TERMS["Time in market"])
    c2.metric(
        "Trades", f"{strat['trades']:,}",
        help=f"{TERMS['Trade']} That's about {strat['trades'] / years_tested:.0f} trades a year.",
    )
    c3.metric(
        "Strategy total return", f"{strat['total_return']:.0%}",
        help=f"Overall gain or loss over the whole test period ({test_period}), after costs.",
    )
    c4.metric(
        "Buy & hold total return", f"{market['total_return']:.0%}",
        help=TERMS["Buy & hold"],
    )

    st.subheader("How much do trading costs matter?")
    st.caption(
        "Each point re-runs the same backtest with a different cost per trade (x-axis). Where the gold "
        "line drops below the grey one, costs have eaten the model's advantage. The dotted line marks "
        "the cost currently set in the sidebar."
    )
    costs = np.arange(0, 20.01, 0.25)
    sweep = pd.DataFrame(
        {
            "Strategy": [performance(strategy_returns(oos, threshold, c)[0])["cagr"] for c in costs],
            "Buy & hold": [performance(buy_and_hold(oos["return"], c))["cagr"] for c in costs],
        },
        index=costs,
    )
    beaten = sweep.index[sweep["Strategy"] <= sweep["Buy & hold"]]
    cost_fig = go.Figure()
    cost_fig.add_scatter(x=costs, y=sweep["Buy & hold"], name="Buy & hold gold", line=dict(color=GREY))
    cost_fig.add_scatter(x=costs, y=sweep["Strategy"], name=f"{model_name} strategy", line=dict(color=GOLD))
    cost_fig.add_vline(x=cost_bps, line_dash="dot", annotation_text="current setting")
    cost_fig.update_layout(height=320, xaxis_title="Cost per trade (bps)", yaxis_title="CAGR",
                           yaxis_tickformat=".0%", margin=dict(t=30, b=10), hovermode="x unified",
                           legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(cost_fig, width="stretch")
    if sweep["Strategy"].iloc[0] <= sweep["Buy & hold"].iloc[0]:
        st.caption("The strategy trails buy & hold even with zero trading costs.")
    elif len(beaten):
        st.caption(f"The strategy's edge over buy & hold disappears at about **{beaten[0]:g} bps** per trade.")
    else:
        st.caption("The strategy stays ahead of buy & hold across this whole cost range.")

# ---------- model comparison ----------
with tab_models:
    st.markdown(
        f"Every model is trained and tested on identical walk-forward folds "
        f"(threshold {threshold:.2f}, costs {cost_bps:g} bps)."
    )
    st.caption(
        "**How to read this:** a useful model needs *both* accuracy clearly above the baseline row "
        "*and* a higher Sharpe ratio than the buy & hold row. Hover a column header for its definition."
    )
    all_oos = {name: get_predictions(name, first_test_year, windows) for name in MODEL_NAMES}
    table = format_table(summarize(all_oos, threshold, cost_bps)).rename_axis("Model").reset_index()
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", help="Open \"What are these models?\" below the table."),
            **{col: st.column_config.TextColumn(col, help=text) for col, text in COLUMN_HELP.items()},
        },
    )
    with st.expander("What are these models?"):
        st.markdown("\n".join(f"- **{name}**: {text}" for name, text in MODEL_INFO.items()))
    st.caption(
        "The majority-class baseline always predicts 'up' when gold rose on most training days, "
        "which makes it identical to buy & hold."
    )

# ---------- per year ----------
with tab_years:
    st.caption(
        "**How to read this:** a genuinely skilful model should beat the 50% coin-flip line (dotted) "
        "in most years, not just a lucky few. **Left:** the model's accuracy each year (bars) next to "
        "how often gold actually rose (line). **Right:** each calendar year's return after costs, "
        "strategy vs. buy & hold."
    )
    year = oos.index.year
    correct = (oos["proba_up"] > threshold).astype(int) == oos["target"]
    yearly = pd.DataFrame(
        {
            "Accuracy": correct.groupby(year).mean(),
            "Up-day share": oos["target"].groupby(year).mean(),
            "Strategy": (1 + strat_daily).groupby(year).prod() - 1,
            "Buy & hold": (1 + market_daily).groupby(year).prod() - 1,
        }
    )
    left, right = st.columns(2)
    acc_fig = go.Figure()
    acc_fig.add_bar(x=yearly.index, y=yearly["Accuracy"], name="Model accuracy", marker_color=GOLD)
    acc_fig.add_scatter(x=yearly.index, y=yearly["Up-day share"], name="Share of up days",
                        mode="lines+markers", line=dict(color=GREY))
    acc_fig.add_hline(y=0.5, line_dash="dot")
    acc_fig.update_layout(title="Accuracy by year", yaxis_tickformat=".0%", yaxis_range=[0.35, 0.65],
                          height=380, margin=dict(b=10), legend=dict(orientation="h", y=-0.15))
    left.plotly_chart(acc_fig, width="stretch")

    ret_fig = go.Figure()
    ret_fig.add_bar(x=yearly.index, y=yearly["Buy & hold"], name="Buy & hold gold", marker_color=GREY)
    ret_fig.add_bar(x=yearly.index, y=yearly["Strategy"], name="Strategy", marker_color=GOLD)
    ret_fig.update_layout(title="Return by year (after costs)", yaxis_tickformat=".0%", barmode="group",
                          height=380, margin=dict(b=10), legend=dict(orientation="h", y=-0.15))
    right.plotly_chart(ret_fig, width="stretch")
    st.caption("2019 is a partial year: the dataset ends on 2 Sep 2019.")

# ---------- methodology ----------
with tab_method:
    st.markdown(
        f"""
**Data.** Daily LBMA gold price fixes (USD, GBP, EUR; AM and PM), Jan 2001 – Sep 2019.
The last trading days before Christmas and New Year have no PM fix; they are dropped *before*
returns are computed, so the next day's return is measured from the last real fix.

**Target.** Whether today's USD PM fix is above yesterday's (up = 1, down/flat = 0).

**Features.** Every feature for day *t* uses information up to day *t-1* only:
lagged returns (1, 2, 3, 5 days), price vs. short- and long-term moving average, rolling volatility,
14-day RSI, and yesterday's EUR/USD move (implied by gold's USD and EUR prices; a weaker dollar
tends to lift gold). Day *t*'s AM fix is deliberately excluded because it is published after the
decision point. A unit test perturbs future prices and asserts no feature changes.

**Validation.** Expanding-window walk-forward: for each year *Y*, train on everything before *Y*,
predict *Y*. No random shuffling, no tuning on the test period.

**Strategy.** Hold gold for the day when P(up) exceeds the threshold, otherwise hold cash
(0% return). Each position change pays the transaction cost. Compared against buy & hold.

**Why the baseline matters.** Gold rose on {clf['up_day_rate']:.0%} of days in this test period,
so {clf['up_day_rate']:.0%} accuracy is what "always say up" achieves. A model is only useful if it beats that by more than the
statistical noise band, and if its trading edge survives realistic costs.
"""
    )

# ---------- glossary ----------
with tab_glossary:
    query = st.text_input(
        "Search terms", placeholder="e.g. Sharpe, drawdown, walk-forward…",
        help="Matches term names and definitions.",
    ).strip().lower()
    matches = {term: text for term, text in TERMS.items() if query in term.lower() or query in text.lower()}
    if not matches:
        st.info(f"No terms match \"{query}\". Try a shorter word.")
    else:
        st.caption(f"Showing {len(matches)} of {len(TERMS)} terms.")
    for term, text in sorted(matches.items(), key=lambda item: item[0].lower()):
        st.markdown(f"**{term}**  \n{text}")
