import pickle

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout, SpatialDropout2D, GaussianNoise
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from sklearn.metrics import roc_auc_score

from data_gen import augment_data


with open('data/train_data.pkl', 'rb') as f:
    x_train, train_labels = pickle.load(f)
y_train = to_categorical(train_labels)

with open('data/test_data.pkl', 'rb') as f:
    x_test, test_labels = pickle.load(f)
y_test = to_categorical(test_labels)

# x_train, y_train = augment_data(x_train, y_train, 0.03)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='constant'
)

datagen.fit(x_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(x_test)

model = Sequential([
    Conv2D(10, kernel_size=(5, 5), activation='relu', data_format='channels_last',
           input_shape=(34, 26, 29),
        #    kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01)
    ),
    # GaussianNoise(0.03),
    MaxPooling2D(pool_size=(2, 2)),
    # SpatialDropout2D(0.3),
    Dropout(0.3),
    Conv2D(10, kernel_size=(4, 3), activation='relu',
           kernel_regularizer=l2(0.005), activity_regularizer=l1(0.005)
    ),
    MaxPooling2D(pool_size=(2, 2)),
    # GaussianNoise(0.03),
    # SpatialDropout2D(0.4),
    # Dropout(0.3),
    Flatten(),
    Dense(5, activation='relu',
          kernel_regularizer=l2(0.01)
    ),
    Dense(2, activation='sigmoid')
])

adam = Adam(lr=0.001)
# sgd = SGD(lr=0.01, nesterov=True)
model.compile(optimizer=adam, loss='binary_crossentropy',
              metrics=['binary_accuracy'])

class_weights = {0: 1, 1: 4}
model.fit_generator(datagen.flow(x_train, y_train, batch_size=128, shuffle=True),
                    steps_per_epoch=len(x_train) / 128, epochs=30,
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
