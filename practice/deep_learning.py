from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


model = Sequential([LSTM(), Dense()])
