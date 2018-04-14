import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

x_train = np.load('X_train_pca.npy')
y_train = np.load('Y_train_pca.npy')
x_test = np.load('X_test_pca.npy')
y_test = np.load('Y_test_pca.npy')

model = Sequential()
model.add(Dense(128, activation='relu', kernel_initializer='normal', input_dim=240))
model.add(Dense(2, activation='softmax', kernel_initializer='normal'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=200,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
