import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": 365}
data = requests.get(url, params=params).json()

prices = data["prices"]
df = pd.DataFrame(prices, columns=["timestamp", "price"])
df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df[["Date", "price"]]
df.set_index("Date", inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["price"]])


SEQ_LEN = 10
x_all, y_all = create_sequences(scaled_data, SEQ_LEN)


split_idx = int(len(x_all) * 0.8)
x_train, x_test = x_all[:split_idx], x_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]



model = Sequential()
model.add(GRU(54, activation='tanh', input_shape=(SEQ_LEN, 1)))
# model.add(Dropout(0.2))
# model.add(GRU(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


checkpoint = ModelCheckpoint(
    filepath='model2.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    callbacks=[checkpoint]
)

best_model = load_model('model.h5')
y_pred = best_model.predict(x_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n==== Kết quả dự báo ====")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²   :", round(r2, 4))

# =========================
# 9. Vẽ biểu đồ giá thực tế vs dự đoán
# =========================
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(y_test):], y_test_inv, label='Giá thực tế')
plt.plot(df.index[-len(y_test):], y_pred_inv, label='Dự báo (GRU)', linestyle='--', color='orange')
plt.xlabel("Ngày")
plt.ylabel("Giá Bitcoin (USD)")
plt.title("Dự báo giá Bitcoin bằng GRU (Best Model)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# =========================
# 10. Vẽ biểu đồ Loss huấn luyện
# =========================
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Biểu đồ Loss theo Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
