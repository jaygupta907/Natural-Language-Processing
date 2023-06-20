
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from datetime import timedelta, date

stock_data = pd.read_csv("ADCON.BO.csv")
stock_data = stock_data.dropna()
date_list = pd.to_datetime(stock_data['Date'])
days_used = 200

plt.figure(figsize=(15, 8))
plt.title('Stock Prices History')
plt.plot(date_list, stock_data['Close'])
plt.xlabel('Date')
plt.ylabel('Prices ($)')
plt.show()

close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))

train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(days_used, len(train_data)):
    x_train.append(train_data[i-200:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=10)

test_data = scaled_data[training_data_len-200:, :]
n = len(test_data)
predictions = []
past_prediction = days_used
predict_days = 200
while (1):
    x_test = test_data[past_prediction-days_used:past_prediction, 0]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
    prediction = model.predict(x_test)
    test_data = np.concatenate((test_data, prediction), axis=0)
    prediction = scaler.inverse_transform(prediction)
    predictions.append(prediction.item())
    past_prediction += 1
    if (past_prediction >= n+predict_days):
        break


date_train = date_list[:training_data_len]
date_test = date_list[training_data_len:]
date_future = []
for i in range(predict_days):
    date_next = date.today()+timedelta(days=i)
    date_future = np.append(date_future, date_next)
date_future = pd.Series(date_future)
date_future = pd.to_datetime(date_future)
date_future = pd.concat([date_test, date_future])


data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(date_train, train)
plt.plot(date_test, validation['Close'])
plt.plot(date_future, predictions)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
