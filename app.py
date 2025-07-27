import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ==============================
# 1. Load mô hình đã huấn luyện
# ==============================
model = load_model('model.h5')

# ==============================
# 2. Tải dữ liệu Bitcoin 10 ngày gần nhất
# ==============================
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

# ==============================
# 3. Tiền xử lý dữ liệu
# ==============================
SEQ_LEN = 10
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(df[["price"]])

# Lấy chuỗi 10 ngày gần nhất để dự đoán ngày tiếp theo
last_seq = scaled_prices[-SEQ_LEN:]
x_input = np.expand_dims(last_seq, axis=0)  # (1, 10, 1)

# ==============================
# 4. Dự đoán giá tiếp theo
# ==============================
pred_scaled = model.predict(x_input)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

# ==============================
# 5. Giao diện Streamlit
# ==============================
st.set_page_config(page_title="Dự báo giá Bitcoin", page_icon="₿")
st.title("📈 Dự báo giá Bitcoin ngày kế tiếp")

st.write("### 🔍 Dữ liệu 10 ngày gần nhất:")
st.line_chart(df["price"])

st.write("### 📊 Giá Bitcoin dự đoán cho ngày kế tiếp là:")
st.success(f"💰 {pred_price:,.2f} USD")

st.write(f"Ngày dự đoán: `{(df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')}`")