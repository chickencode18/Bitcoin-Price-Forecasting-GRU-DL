import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

model = load_model('model.h5')

def load_recent_data(days=20):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["Date", "price"]]
    df.set_index("Date", inplace=True)
    return df

df = load_recent_data()

SEQ_LEN = 10
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(df[["price"]])
last_seq = scaled_prices[-SEQ_LEN:]
x_input = np.expand_dims(last_seq, axis=0)

pred_scaled = model.predict(x_input)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

st.set_page_config(page_title="Bitcoin Price Prediction", page_icon="₿")
st.title("📈 Bitcoin Price Prediction for Next Day")

st.write("### 🔍 Last 10 Days Data:")
st.line_chart(df["price"])

st.write("### 📊 Predicted Bitcoin Price for Next Day:")
st.success(f"💰 {pred_price:,.2f} USD")

st.write(f"Predicted Date: `{(df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')}`")
