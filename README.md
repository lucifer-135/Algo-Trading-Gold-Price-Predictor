# 💰 Algo-Trading Gold Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://algo-trading-gold-price-predictor.streamlit.app/)
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

## 📓 Exploratory Data Analysis (EDA)

The repository includes a Jupyter Notebook (`gold_price_prediction.ipynb`) that documents the research and prototyping phase:
* **Data Visualization:** Detailed plots of price history, return distributions, and correlation heatmaps.
* **Model Comparison:** Initial experiments comparing Linear Regression vs. Random Forest performance.
* **Feature Importance:** Analysis of which indicators (Volatility, Momentum, etc.) had the most impact on price direction.
* *Note: This notebook serves as the "lab report" showing the rationale behind the final model selection.*

## 🛠️ Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib

## 🚀 How to Run Locally

### Prerequisites
* Python 3.8 or higher

### Step 1: Clone the Repository
```bash
git clone https://github.com/lucifer-135/Algo-Trading-Gold-Price-Predictor.git
cd Algo-Trading-Gold-Price-Predictor
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the App
```bash
streamlit run app.py
```
