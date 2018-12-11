from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Dropout(0.4),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(10, activation='softmax')
])

adam = Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=12, batch_size=32)

score = model.evaluate(x_test, y_test)

print("Test accuracy: ", score[1])
preds = model.predict(x_test)
auc = roc_auc_score(y_test, preds)
print(f"AUROC: {auc}")
