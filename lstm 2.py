#Import the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import pandas_datareader as dr
import date as dt
#Import Dataset
download_source = (r'D:\Project\TWTR.csv')

start = dt.datetime(2015,5,5)
end = dt.datetime.today()

df = pdr.get_data_yahoo('TWTR',start,end)
print(df.tail())

df.to_csv(download_source)


#Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(download_source,test_size = 0.2,shuffle=False)

open_train_data = train_data.iloc[:,2:3].values

#Data Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
open_scaled_data = scaler.fit_transform(open_train_data)

#Convert Training Data to Right Shape
features_set = []
labels = []
for i in range(60, 1161):
    features_set.append(open_scaled_data[i-60:i, 0])
    labels.append(open_scaled_data[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

#Training The LSTM
model = Sequential()

#Creating LSTM and Dropout Layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

#Creating Dense Layer
model.add(Dense(units = 1))

#Model Compilation
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

#Algorithm Training
model.fit(features_set, labels, epochs = 100, batch_size = 32)

#Testing our LSTM
open_test_data = test_data.iloc[:,2:3].values

#Converting Test Data to Right Format
total_data = pd.concat((train_data['Open'], test_data['Open']), axis=0)

total_data

test_inputs = total_data[len(total_data) - len(test_data) - 60:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

test_features = []
for i in range(60, 504):
    test_features.append(test_inputs[i-60:i, 0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

#Making Predictions
predictions = model.predict(test_features)

predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10,6))
plt.plot(open_test_data, color='blue', label='Actual IBM Stock Price')
plt.plot(predictions , color='red', label='Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.show()
