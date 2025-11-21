import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import io

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import matplotlib.pyplot as plt

# ——— Config ———
MODEL_PATH    = Path("best_stock_model.h5")
DEFAULT_TICK  = "GOOG"
DEFAULT_START = "2012-01-01"

@st.cache_data(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    # compute features
    for w in (10, 50, 100, 200):
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

# ——— UI Setup ———
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("📈 Stock Market Predictor")

# Initialize session state for control flow
if 'run_pred' not in st.session_state:
    st.session_state.run_pred = False

# Sidebar controls
with st.sidebar:
    ticker     = st.text_input("Ticker Symbol", DEFAULT_TICK).upper()
    start_date = st.date_input("Start Date", pd.to_datetime(DEFAULT_START))
    end_date   = st.date_input("End Date", pd.to_datetime("today"))
    n_steps    = st.slider("Sequence Length", 50, 200, 100)
    if st.button("Run Prediction"):
        st.session_state.run_pred = True

# Guard: only proceed after clicking "Run Prediction"
if not st.session_state.run_pred:
    st.info("Adjust parameters in the sidebar and click **Run Prediction**")
    st.stop()

# Load model and data
model = load_model()
df    = fetch_data(ticker, start_date, end_date)

# Raw data display
st.subheader("Historical Data (OHLCV)")
st.dataframe(df.set_index("Date")[['Open','High','Low','Close','Volume']])

# Split & scale
split_idx    = int(len(df) * 0.8)
train_df     = df.iloc[:split_idx]
test_df      = df.iloc[split_idx - n_steps:]
scaler       = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[["Close","MA_10","MA_50","MA_100","MA_200","RSI_14"]])
test_scaled  = scaler.transform(test_df[["Close","MA_10","MA_50","MA_100","MA_200","RSI_14"]])

# Build sequences
def make_seq(arr, steps):
    X = []
    for i in range(steps, len(arr)):
        X.append(arr[i-steps:i])
    return np.array(X)

X_test = make_seq(test_scaled, n_steps)
y_true = test_scaled[n_steps:, 0] * (1.0 / scaler.scale_[0])

# Predict
y_pred = model.predict(X_test).flatten() * (1.0 / scaler.scale_[0])

# Charts: Price & Indicators
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price & Moving Averages")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Date'], df['Close'], label='Close', color='k')
    for w, c in zip((10, 50, 100), ('c', 'm', 'y')):
        ax.plot(df['Date'], df[f'MA_{w}'], label=f'MA {w}', color=c)
    ax.set_xlabel('Date'); ax.set_ylabel('Price')
    ax.legend(); st.pyplot(fig)

with col2:
    st.subheader("RSI (14)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Date'], df['RSI_14'], label='RSI_14', color='b')
    ax.axhline(70, color='r', linestyle='--')
    ax.axhline(30, color='g', linestyle='--')
    ax.set_ylabel('RSI'); st.pyplot(fig)

# Prediction vs Actual
st.subheader("Model Prediction vs Actual")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Date'].iloc[split_idx:], y_true, label='Actual', color='black')
ax.plot(df['Date'].iloc[split_idx:], y_pred, label='Predicted', color='r')
ax.set_xlabel('Date'); ax.set_ylabel('Price')
ax.legend(); st.pyplot(fig)

# Metrics display
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
st.subheader("Model Performance")
st.write(f"**MSE:** {mse:.2f} **MAE:** {mae:.2f} **R²:** {r2:.2f}")

# Next-Day Prediction Button
st.subheader("Next-Day Forecast")
if st.button("Predict Next-Day Close"):
    # determine next trading date (approximate)
    last_date = df['Date'].max()
    next_date = last_date + pd.Timedelta(days=1)
    # prepare latest window
    last_window = df[["Close","MA_10","MA_50","MA_100","MA_200","RSI_14"]].tail(n_steps)
    last_scaled = scaler.transform(last_window)
    X_next      = last_scaled.reshape(1, n_steps, last_scaled.shape[1])
    y_next_scaled = model.predict(X_next).flatten()[0]
    data_min    = scaler.data_min_[0]
    data_range  = scaler.data_range_[0]
    y_next      = y_next_scaled * data_range + data_min
    st.markdown(f"### 📈 Predicted Close for {next_date.date()}: **${y_next:.2f}**")

# Sample predictions table
st.subheader("Sample Predictions")
pred_dates = df['Date'].iloc[split_idx:].reset_index(drop=True)
results = pd.DataFrame({
    'Date': pred_dates,
    'Actual_Close': y_true,
    'Predicted_Close': y_pred
}).set_index('Date')
st.dataframe(results.tail(10))

# Model info
st.subheader("Model Information")
total_params = model.count_params()
st.write(f"**Total Parameters:** {total_params}")
with st.expander("View Model Summary"):
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    st.text(buf.getvalue())
