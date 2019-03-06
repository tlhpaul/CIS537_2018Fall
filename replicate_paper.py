import pickle

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout, SpatialDropout2D, GaussianNoise
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

#from data_gen import augment_data


with open('../train_data.pkl', 'rb') as f:
    x_train, train_labels = pickle.load(f)
y_train = to_categorical(train_labels)

with open('../test_data.pkl', 'rb') as f:
    x_test, test_labels = pickle.load(f)
y_test = to_categorical(test_labels)

datagen = ImageDataGenerator()
datagen.fit(x_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(x_test)

model = Sequential([
    Conv2D(10, kernel_size=(5, 5), activation='tanh',
           data_format='channels_last', input_shape=(34, 26, 29)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(10, kernel_size=(4, 3), activation='tanh'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(5, activation='tanh'),
    Dense(2, activation='sigmoid')
])

sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy',
              metrics=['binary_accuracy'])

callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
                         verbose=0, mode='auto', baseline=0.7)

class_weights = {0: 1, 1: 4}
model.fit_generator(datagen.flow(x_train, y_train, batch_size=1, shuffle=True),
                    callbacks=[callback],
                    steps_per_epoch=len(x_train), epochs=100,
                    class_weight=class_weights,
                    validation_data=val_datagen.flow(x_test, y_test),
                    nb_val_samples=x_test.shape[0])

score = model.evaluate(x_test, y_test)

print("Weighted test accuracy: ", score[1])
preds = model.predict(x_test)
auc = roc_auc_score(y_test, preds)
print(model.summary())
print(f"AUROC: {auc}")

model.save('model/most_recent.h5')
