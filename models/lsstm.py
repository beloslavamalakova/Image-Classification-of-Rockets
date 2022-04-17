from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM


model = Sequential();
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1])), 1);
model.add(Dropout(0.2));
model.add(LSTM(units=50, return_sequences=True));
model.add(Dropout(0.2));
model.add(LSTM(units=50));
model.add(Dropout(0.2));

model.add(Dense(units=1));
model.compile(optimizer='adam', loss='mean_squared_error');
model.fit(X_train, y_train, epochs=100, batch_size=32);


