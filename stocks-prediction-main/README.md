# Stock Pro

**An interactive Streamlit app for deep-learning–based stock price analysis and next-day forecasting.**

---

##  Overview

Stock Prophet is an end-to-end Python application that:

* Fetches 10+ years of OHLCV data using the Yahoo Finance API (`yfinance`).
* Calculates technical indicators (moving averages, RSI).
* Trains a stacked LSTM model for time-series prediction.
* Presents results via an interactive Streamlit dashboard with Plotly charts.

---

##  Features

* **Data & Indicators**

  * 10-year historical price/volume (OHLCV)
  * Simple Moving Averages (10, 50, 100, 200 days)
  * 14-day Relative Strength Index (RSI)

* **Model Architecture**

  * Stacked LSTM layers (64 → 128 units) with Dropout
  * EarlyStopping & ModelCheckpoint callbacks

* **Interactive Dashboard**

  * **Overview**: key metrics (current price, 52‑week high/low, RSI, MA values)
  * **Analysis**: Plotly candlestick charts overlaid with SMAs, and RSI panel with overbought/oversold bands
  * **Prediction**: Actual vs. predicted close price plot, performance metrics (RMSE, MAPE, R²), and a one‑click "Next-Day Forecast"
  * **Model Info**: parameter count and expandable model summary in code format
  * Responsive horizontal navigation menu

---

## ⚙️ Tech Stack

* Python 3.8+
* TensorFlow & Keras (LSTM)
* scikit-learn
* Streamlit
* Plotly
* pandas, numpy
* Yahoo Finance API (`yfinance`)

---

## 📥 Installation

```bash
# Clone this repository
git clone https://github.com/your-username/stock-prophet.git
cd stock-prophet

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶ Usage

```bash
# Run the Streamlit app\streamlit run app.py
```

1. Adjust ticker symbol, date range, and sequence length in the sidebar.
2. Navigate between tabs: **Overview**, **Analysis**, **Prediction**, **Model Info**.
3. In **Prediction**, click **Next‑Day Forecast** for tomorrow’s closing price.

---

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

* Submit a GitHub issue to report bugs or suggest enhancements.
* Fork the repo and open a pull request with your improvements.

---


