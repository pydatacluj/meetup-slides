import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, LSTM, RepeatVector
from keras.models import Model, Sequential
from keras import regularizers

optimizer = "Nadam"
steps = 30
sample_size = 20
latent_dim = 3
batch_size = 64
epochs = 50
dropout_rate = 0

np.random.seed(1234)

inputs = Input(shape=(steps, sample_size,))
encoded = LSTM(latent_dim, activation='relu', name='latent', dropout=dropout_rate, recurrent_dropout=dropout_rate)(inputs)
outputs = RepeatVector(steps)(encoded)
outputs = LSTM(sample_size, return_sequences=True)(outputs)

ae = Model(input=inputs, output=outputs)

encoder = Model(inputs, encoded)

temp = pd.read_csv("temperatures.csv").values
pct = 0.6
idx = int(len(temp) * pct)

x_train = temp[:idx,1:]
x_test = temp[(idx+1):,1:]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

data = []
for i in range(len(x_train)-steps):
	data.append(x_train[i:(i+steps),])
x_train = np.array(data)

data = []
for i in range(len(x_test)-steps):
	data.append(x_test[i:(i+steps),])
x_test = np.array(data)


y_train = x_train
y_test = x_test

ae.compile(optimizer=optimizer, loss="mae")
ae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

z_test = encoder.predict(x_train)
np.savetxt("outputs/lstm_temperatures_activations-train.csv", z_test, delimiter=",", fmt='%f')
z_test = encoder.predict(x_test)
np.savetxt("outputs/lstm_temperatures_activations-test.csv", z_test, delimiter=",", fmt='%f')
