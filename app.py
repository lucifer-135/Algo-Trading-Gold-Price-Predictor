import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#CONFIGURATION
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("💰 Gold Price Prediction Using ML")
st.markdown("""
This app uses a **Random Forest Classifier** to predict whether Gold prices will go **UP** or **DOWN**.
""")

#LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("gold_price.csv", parse_dates=True, index_col='Date')
    df['Return'] = df['USD (PM)'].pct_change() * 100
    df['Lagged_Return'] = df['Return'].shift()
    df = df.dropna()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'gold_price.csv' not found.")
    st.stop()

#SIDEBAR CONTROLS
st.sidebar.header("⚙️ Model Parameters")
split_year = st.sidebar.slider("Training Cutoff Year", 2015, 2018, 2018)
ma_window = st.sidebar.slider("Moving Average Window (In Days)", 2, 30, 3)
volatility_window = st.sidebar.slider("Volatility Window (In Days)", 2, 30, 7)

#FEATURE ENGINEERING
df['SMA'] = df['Lagged_Return'].rolling(window=ma_window).mean()
df['Volatility'] = df['Lagged_Return'].rolling(window=volatility_window).std()
df['Momentum'] = df['Lagged_Return'] - df['Lagged_Return'].shift(3)
df = df.dropna()

#TRAIN MODEL
train = df.loc[df.index.year <= split_year]
test = df.loc[df.index.year > split_year]

features = ['Lagged_Return', 'SMA', 'Volatility', 'Momentum']
X_train = train[features]
X_test = test[features]

#Create Binary Targets (1 = Up, 0 = Down/Flat)
y_train = np.where(train['Return'] > 0, 1, 0)
y_test = np.where(test['Return'] > 0, 1, 0)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#DISPLAY METRICS
accuracy = accuracy_score(y_test, predictions)

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy:.2%}", help="Percentage of time the model correctly predicted direction.")
col2.metric("Training Days", len(train))
col3.metric("Test Days", len(test))

#VISUALIZATIONS
st.subheader("Cumulative Profit Strategy (Backtest)")

# Logic: Since 'predictions' is already 1 (Buy) or 0 (Sit out), we use it directly
test['Predicted_Signal'] = predictions 
test['Strategy_Return'] = test['Return'] * test['Predicted_Signal']

# Calculate Cumulative Returns
test['Cumulative_Market'] = (1 + test['Return'] / 100).cumprod()
test['Cumulative_Strategy'] = (1 + test['Strategy_Return'] / 100).cumprod()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(test.index, test['Cumulative_Market'], label='Market (Buy & Hold)', color='gray', alpha=0.5)
ax.plot(test.index, test['Cumulative_Strategy'], label='Random Forest Strategy', color='green')
ax.set_ylabel("Growth of $1 Investment")
ax.legend()
st.pyplot(fig)

# Final verdict
final_market = test['Cumulative_Market'].iloc[-1]
final_strategy = test['Cumulative_Strategy'].iloc[-1]

if final_strategy > final_market:
    st.success(f"🚀 The Random Forest beat the market!")
else:

    st.warning("⚠️ The market performed better this time. Try adjusting the sliders.")
