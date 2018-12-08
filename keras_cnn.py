import pickle

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.utils import to_categorical


with open('data/train_data.pkl', 'rb') as f:
    train_data, train_labels = pickle.load(f)
y_train = to_categorical(train_labels)

with open('data/test_data.pkl', 'rb') as f:
    test_data, test_labels = pickle.load(f)
y_test = to_categorical(test_labels)

model = Sequential([
    Conv2D(64, kernel_size=3, activation='relu', input_shape=(34, 26, 29)),
    Conv2D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, y_train, validation_data=(test_data, y_test), epochs=5)
