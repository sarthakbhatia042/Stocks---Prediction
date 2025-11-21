import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import io

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px

# ——— Config ———
MODEL_PATH    = Path("best_stock_model.h5")
DEFAULT_TICK  = "GOOG"
DEFAULT_START = "2012-01-01"

# ——— Streamlit Setup ———
st.set_page_config(page_title=" Stock Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
.block-container { padding: 1rem 2rem; }
.stButton>button { background-color: #1E3A8A; color: #fff; }
.nav-link-icon {
    color: #1E3A8A !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Indicators
    for w in (10, 50, 100, 200):
        df[f"MA_{w}"] = df["Close"].rolling(window=w).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

# Sidebar controls
ticker = st.sidebar.text_input("Ticker", DEFAULT_TICK).upper()
start_date = st.sidebar.date_input("Start", pd.to_datetime(DEFAULT_START))
end_date = st.sidebar.date_input("End", pd.to_datetime("today"))
n_steps = st.sidebar.slider("Sequence Length", 30, 200, 60)

# Top navigation with custom styling
st.markdown(
    """
    <div style='background-color:#f0f2f6;padding:10px;border-radius:8px;margin-bottom:20px;'>
    """, unsafe_allow_html=True)
selection = option_menu(
    menu_title=None,
    options=["Overview", "Analysis", "Prediction", "Model Info"],
    icons=["house", "bar-chart", "magic", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0px", "background-color": "#f0f2f6"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#555"},
        "nav-link-selected": {"background-color": "#1E3A8A", "color": "white", "font-weight": "bold"},
        "icon": {"color": "#1E3A8A", "font-size": "20px"},
    },
    key="main_nav"
)
st.markdown("""
    </div>
    """, unsafe_allow_html=True)

# Load model & data
model = load_model()
df = fetch_data(ticker, start_date.isoformat(), end_date.isoformat())

# Shared split and scaling
split_idx = int(len(df) * 0.8)
feature_cols = ["Close"] + [f"MA_{w}" for w in (10,50,100,200)] + ["RSI_14"]
feature_cols = [c for c in feature_cols if c in df.columns]
idx = feature_cols.index("Close")
scaler = MinMaxScaler()
train = scaler.fit_transform(df[feature_cols].iloc[:split_idx])
test = scaler.transform(df[feature_cols].iloc[split_idx-n_steps:])
X_test = np.array([test[i-n_steps:i] for i in range(n_steps, len(test))])
y_true = test[n_steps:, idx] * scaler.data_range_[idx] + scaler.data_min_[idx]
y_pred = model.predict(X_test).flatten() * scaler.data_range_[idx] + scaler.data_min_[idx]

if selection == "Overview":
    st.title(f"📈 {ticker} Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        curr = float(df["Close"].iloc[-1])
        change_pct = float(df["Close"].pct_change().iloc[-1] * 100)
        st.metric("Current Price", f"${curr:.2f}", f"{change_pct:.2f}%")
        st.metric("52W High", f"${float(df['High'].max()):.2f}")
    with col2:
        st.metric("52W Low", f"${float(df['Low'].min()):.2f}")
        st.metric("RSI (14)", f"{float(df['RSI_14'].iloc[-1]):.2f}")
    with col3:
        ma50 = float(df['MA_50'].iloc[-1]) if 'MA_50' in df else None
        ma200 = float(df['MA_200'].iloc[-1]) if 'MA_200' in df else None
        st.metric("MA(50)", f"${ma50:.2f}" if ma50 else "N/A")
        st.metric("MA(200)", f"${ma200:.2f}" if ma200 else "N/A")
    st.markdown("---")
    st.dataframe(df.tail(10).set_index("Date"))

elif selection == "Analysis":
    st.title(f"🔍 {ticker} Analysis")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    for w,color in [(10,'orange'), (50,'blue')]:
        col = f"MA_{w}"
        if col in df:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[col], mode='lines', name=col, line=dict(color=color)))
    fig.update_layout(title='Candlestick & MAs', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], mode='lines', name='RSI_14'))
    fig2.add_hline(y=70, line_dash='dash', annotation_text='Overbought')
    fig2.add_hline(y=30, line_dash='dash', annotation_text='Oversold')
    fig2.update_layout(title='RSI (14)')
    st.plotly_chart(fig2, use_container_width=True)

elif selection == "Prediction":
    st.title(f"🔮 {ticker} Prediction")
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=df['Date'].iloc[split_idx:], y=y_true, mode='lines', name='Actual'))
    pred_fig.add_trace(go.Scatter(x=df['Date'].iloc[split_idx:], y=y_pred, mode='lines', name='Predicted'))
    pred_fig.update_layout(title='Actual vs Predicted Close')
    st.plotly_chart(pred_fig, use_container_width=True)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric('RMSE', f'${rmse:.2f}')
    c2.metric('MAPE', f'{mape:.2f}%')
    c3.metric('R²', f'{r2_score(y_true, y_pred):.4f}')
    st.markdown('---')
    if st.button('Next-Day Forecast'):
        last = df[feature_cols].tail(n_steps)
        scaled = scaler.transform(last)
        y_n = model.predict(scaled.reshape(1,n_steps,len(feature_cols))).flatten()[0]
        y_n = y_n * scaler.data_range_[idx] + scaler.data_min_[idx]
        st.success(f'Next-Day Close: ${y_n:.2f}')

else:
    st.title('ℹ️ Model Information')
    st.write(f'**Total Parameters:** {model.count_params()}')
    with st.expander('Model Summary'):
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + '\n'))
        st.code(buf.getvalue(), language='text')
