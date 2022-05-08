import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np

num_emotions = 7
batch_size = 256
steps_per_epoch = 112
epochs = 11
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def split_data():
  # Import fer2013.csv
  with open("fer2013.csv") as file:
      data = file.readlines()
  lines = np.array(data)
  x_train, y_train, x_test, y_test = [], [], [], []

  # Split dataset into training and test sets
  for i in range(1,lines.size):
      emotion, img, usage = lines[i].split(",")
      val = img.split(" ")
      pixels = np.array(val, 'float32')
      emotion = keras.utils.np_utils.to_categorical(emotion, num_emotions)

      if 'Training' in usage:
          y_train.append(emotion)
          x_train.append(pixels)
      elif 'PublicTest' in usage:
          y_test.append(emotion)
          x_test.append(pixels)

  # Cast and normalize data
  x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
  x_train, x_test = np.true_divide(x_train, 255.0), np.true_divide(x_test, 255.0)
  
  # Make sure data is in the right shape
  x_train, x_test = x_train.reshape( (len(x_train),48,48,1) ), x_test.reshape( (len(x_test),48,48,1) )
  print("x_train, y_train, x_test, y_test: ",x_train.shape, y_train.shape, x_test.shape, y_test.shape)
  return x_train, y_train, x_test, y_test

def create_model():
  inputs = Input(shape=(48, 48, 1, ))

  conv = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(inputs)
  conv = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv)
  pool = MaxPooling2D(pool_size=(2,2))(conv)
  dropout = Dropout(0.4)(pool)

  conv = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(dropout)
  pool = MaxPooling2D(pool_size=(2,2))(conv)
  conv = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(pool)
  pool = MaxPooling2D(pool_size=(2,2))(conv)
  dropout = Dropout(0.4)(pool)

  flatten = Flatten()(dropout)
  dense = Dense(1024, activation='relu')(flatten)
  dropout = Dropout(0.5)(dense)
  pred = Dense(7, activation='softmax')(dropout)
  return Model(inputs=inputs, outputs=pred)


def cnn():
  x_train, y_train, x_test, y_test = split_data()
  model = create_model()

  # Use ImageDataGenerator for better generalizability
  datagen = ImageDataGenerator()

  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

  # Train model, save for quick reload later
  model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch, 
            validation_data=datagen.flow(x_test, y_test, batch_size=batch_size))
  model.save('../models/face-emotion.h5')

def test_cnn():
  model = load_model('../models/face-emotion.h5')
  x_train, y_train, x_test, y_test = split_data()
  print("evaluating facial emotion recognition model")
  model.evaluate(x_test, y_test)

cnn()
test_cnn()