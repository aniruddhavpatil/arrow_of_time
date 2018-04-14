import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

# x_train = np.random.random((1000, 100))
# y_train = keras.utils.to_categorical(np.random.randint(4, size=(1000, 1)), num_classes=4)
# x_test = np.random.random((100, 100))
# y_test = keras.utils.to_categorical(np.random.randint(4, size=(100, 1)), num_classes=4)

x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(128, activation='relu', kernel_initializer='normal', input_dim=4000))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax', kernel_initializer='normal'))



# sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=40,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
