import pickle

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout, SpatialDropout2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import set_random_seed

with open('../data/train_data.pkl', 'rb') as f:
    train_data, train_labels = pickle.load(f)
train_classes = to_categorical(train_labels)

with open('../data/test_data.pkl', 'rb') as f:
    test_data, test_labels = pickle.load(f)
test_classes = to_categorical(test_labels)

# Set numpy and TensorFlow random seeds in the hopes of making
# results reproducible. This will not be possible when using a GPU,
# as there may be asynchronous processing for which no random seed
# could account.
set_random_seed(2)
np.random.seed(1)

datagen = ImageDataGenerator()
datagen.fit(train_data)

val_datagen = ImageDataGenerator()
val_datagen.fit(test_data)

# model = Sequential([
#     Conv2D(10, kernel_size=(5, 5), activation='tanh',
#            data_format='channels_last', input_shape=(34, 26, 29)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.2),
#     Conv2D(10, kernel_size=(4, 3), activation='tanh'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.4),
#     Flatten(),
#     Dense(5, activation='tanh'),
#     Dense(2, activation='sigmoid')
# ])

# sgd = SGD(lr=0.01)
# model.compile(optimizer=sgd, loss='binary_crossentropy',
#               metrics=['binary_accuracy'])

model = Sequential([
    Conv2D(10, kernel_size=(3, 3), activation='tanh',
           data_format='channels_last', input_shape=(34, 26, 29)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),
    Conv2D(10, kernel_size=(3, 3), activation='tanh'),
    Dropout(0.4),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(10, activation='tanh'),
    Dense(2, activation='softmax')
])

adam = Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


callback = EarlyStopping(monitor='val_loss', min_delta=-0.1, patience=3,
                         verbose=1, mode='auto', baseline=0.8)

# class_weights = {0: 1, 1: 4}
model.fit_generator(datagen.flow(train_data, train_classes, batch_size=1, shuffle=True),
#                     callbacks=[callback],
                    steps_per_epoch=len(train_data), epochs=30,
#                     class_weight=class_weights,
                    validation_data=val_datagen.flow(test_data, test_classes),
                    nb_val_samples=test_data.shape[0])

score = model.evaluate(test_data, test_classes)

print(model.summary())
print("Test accuracy: ", score[1])
preds = model.predict(test_data)
auc = roc_auc_score(test_classes, preds)
print(f"Test AUROC: {auc}")


# Check AUC on the training data, just to verify that the training data was learned.
score = model.evaluate(train_data, train_classes)

preds = model.predict(train_data)
auc = roc_auc_score(train_classes, preds)
print("Training data accuracy: ", score[1])
print(f"Training AUROC: {auc}")
print(len(train_classes))
model.save('../model/most_recent.h5')
