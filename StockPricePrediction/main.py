import math
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin

# displays the whole data frame
pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

yfin.pdr_override()
plt.style.use('fivethirtyeight')

start = datetime(2015, 1, 1)
end = datetime(2023, 5, 11)

df = pdr.get_data_yahoo('AAPL', start=start, end=end)

data = df.filter(['Close'])

dataset = data.values

training_data_len = math.ceil(len(dataset) * 0.8)

train_data = dataset[0:training_data_len, :]
test_data = dataset[training_data_len - 60:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

print(train_data_scaled.shape, test_data_scaled.shape)

x_train = []
y_train = []

for i in range(60, len(train_data_scaled)):
    x_train.append(train_data_scaled[i-60:i, 0])
    y_train.append(train_data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train te model
model.fit(x_train, y_train, batch_size=10, epochs=50)

# Create the x_test and y_test data sets

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data_scaled)):
    x_test.append(test_data_scaled[i-60:i, 0])

# Convert the data to numpy array
x_test = np.array(x_test)

# Reshape the x_test data set
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)

rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(f'RMSE: {rmse}')

# Plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

#print(valid)
#plt.show()

# Get the quote
apple_quote = pdr.get_data_yahoo('AAPL', start=start, end=end)

# Create a new data frame
new_df = apple_quote.filter(['Close'])

# Get the last 60 days closing price and convert the df to array
last_sixty_days = new_df[-60:].values

# Scale the data
last_sixty_days_scaled = scaler.transform(last_sixty_days)

# Create and append the last 60 days data into the new test array
x_test_sixty_days = []

x_test_sixty_days.append(last_sixty_days_scaled)

# Convert the array to a num py array
x_test_sixty_days = np.array(x_test_sixty_days)

# Reshape the num py array
x_test_sixty_days = np.reshape(x_test_sixty_days, (x_test_sixty_days.shape[0], x_test_sixty_days.shape[1], 1))

# Get the predicted price
predictions_sixty_days = model.predict(x_test_sixty_days)
predictions_sixty_days = scaler.inverse_transform(predictions_sixty_days)

print(predictions_sixty_days)

start = datetime(2023, 5, 12)
end = datetime(2023, 5, 13)

apple_quote = pdr.get_data_yahoo('AAPL', start=start, end=end)
print(apple_quote['Close'])

plt.show()
