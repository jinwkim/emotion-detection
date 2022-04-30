import numpy as np

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr

import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks 

import sklearn
from sklearn.model_selection import train_test_split

batch_size = 23

x = np.load('data/features.npy')
y = np.load('data/emotions.npy')

print(x.shape)
print(y.shape)

# split data into train, test, and validation sets
x_train, x_tosplit, y_train, y_tosplit = train_test_split(x, y, test_size = 0.125, random_state = 1)
x_val, x_test, y_val, y_test = train_test_split(x_tosplit, y_tosplit, test_size = 0.304, random_state = 1)

y_train_class = tf.keras.utils.to_categorical(y_train, 8, dtype = 'int8')
y_val_class = tf.keras.utils.to_categorical(y_val, 8, dtype = 'int8')
y_test_class = tf.keras.utils.to_categorical(y_test, 8, dtype = 'int8')

print(np.shape(x_train))
print(np.shape(x_val))
print(np.shape(x_test))

# creating LTSM (Long Short-Term Memory) model
model = Sequential()
model.add(layers.LSTM(64, return_sequences = True, input_shape = (x.shape[1:3])))
model.add(layers.LSTM(64))
model.add(layers.Dense(8, activation = 'softmax'))
print(model.summary())

# reduce learning rate after 100 epochs without improvement.
rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', 
                                    factor=0.1, patience=100)

model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])

model.fit(x_train, y_train_class, 
    epochs=340, batch_size = batch_size, 
    validation_data = (x_val, y_val_class), 
    callbacks = [rlrop])

model.save('../models/speech_emotion.h5')

# evaluate with validation and testing sets
val_loss, val_acc = model.evaluate(x_val, y_val_class, verbose = 2)
print('validation accuracy', val_acc)
test_loss, test_acc = model.evaluate(x_test, y_test_class, verbose = 2)
print('testing accuracy', test_acc)

