# 💰 Algo-Trading Gold Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://share.streamlit.io/](https://algo-trading-gold-price-predictor.streamlit.app/))
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning application that forecasts gold price trends and simulates algorithmic trading strategies. This project uses a **Random Forest Classifier** to predict market direction (Up/Down) based on technical indicators and visualizes the profitability against a standard "Buy and Hold" strategy via an interactive **Streamlit** dashboard.

---

## 🚀 Features

* **Machine Learning Pipeline:** Trains a Random Forest Classifier on historical financial data to predict daily price movements.
* **Feature Engineering:** Transforms raw time-series data into technical indicators:
    * **SMA (Simple Moving Average):** Captures long-term trends.
    * **Volatility:** Measures market risk/standard deviation.
    * **Momentum:** Tracks the speed of price changes.
* **Algorithmic Backtesting:** Simulates a trading strategy based on model predictions and calculates Cumulative Return vs. Market Baseline.
* **Interactive Dashboard:** A user-friendly web interface allowing real-time parameter tuning (Training Window, Moving Average configurations).

## 🛠️ Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib

## 📂 Project Structure

```bash
├── app.py                # Main application script (Streamlit + ML Logic)
├── gold_price.csv        # Historical dataset (Time-series data)
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
