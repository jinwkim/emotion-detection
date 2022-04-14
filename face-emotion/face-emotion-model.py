import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
steps_per_epoch = 112
epochs = 11
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def plot_emotion_prediction(pred):
    """
    Plots the prediction for each emotion on a bar chart
    :param pred: the predictions for each emotions
    """
    labels = np.arange(len(emotions))
    plt.bar(labels, pred, align='center', alpha=0.5)
    plt.xticks(labels, emotions)
    plt.ylabel('prediction')
    plt.title('emotion')
    plt.show()

def split_data():
  # READ IN KAGGLE DATA
  with open("../data/fer2013.csv") as file:
      data = file.readlines()
  lines = np.array(data)
  x_train, y_train, x_test, y_test = [], [], [], []
  # A. 1) SPLIT DATA INTO TEST AND TRAIN
  for i in range(1,lines.size):
      emotion, img, usage = lines[i].split(",")
      val = img.split(" ")
      pixels = np.array(val, 'float32')
      emotion = keras.utils.np_utils.to_categorical(emotion, num_classes)
      if 'Training' in usage:
          y_train.append(emotion)
          x_train.append(pixels)
      elif 'PublicTest' in usage:
          y_test.append(emotion)
          x_test.append(pixels)
  # A. 2) CAST AND NORMALIZE DATA
  x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
  x_train, x_test = np.true_divide(x_train, 255.0), np.true_divide(x_test, 255.0)
  # A. 3) RESHAPE DATA
  x_train, x_test = x_train.reshape( (len(x_train),48,48,1) ), x_test.reshape( (len(x_test),48,48,1) )
  # print("x_train, y_train, x_test, y_test: ",x_train.shape, y_train.shape, x_test.shape, y_test.shape)
  return x_train, y_train, x_test, y_test

def create_model():
  inputs = Input(shape=(48, 48, 1, ))
  # INSERT LAYERS HERE
  conv = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(inputs)
  conv = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(conv)
  pool = MaxPooling2D(pool_size=(2,2))(conv)
  dropout = Dropout(0.2)(pool)
  conv = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(dropout)
  conv = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv)
  pool = MaxPooling2D(pool_size=(2,2))(conv)
  dropout = Dropout(0.2)(pool)
  conv = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(dropout)
  conv = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(conv)
  pool = MaxPooling2D(pool_size=(2,2))(conv)
  dropout = Dropout(0.4)(pool)
  flatten = Flatten()(dropout) # need to flatten into 1-D array before dense
  dense = Dense(1024, activation='relu')(flatten)
  pred = Dense(7, activation='softmax')(dense)
  return Model(inputs=inputs, outputs=pred)

def cnn():
  x_train, y_train, x_test, y_test = split_data()
  model = create_model()

  # C. 1) DATA BATCH PROCESS
  datagen = ImageDataGenerator()

  # C. 2) COMPILE MODEL
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

  # C. 3) TRAIN AND SAVE MODEL
  model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch, 
            validation_data=datagen.flow(x_test, y_test, batch_size=batch_size))
  model.save('../models/model.h5')

def test_cnn():
  model = load_model('../models/model.h5')
  x_train, y_train, x_test, y_test = split_data()
  model.evaluate(x_test, y_test)
  img = image.load_img("../data/jinkim.png", color_mode = "grayscale", target_size=(48, 48))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis = 0)
  x /= 255
  custom = model.predict(x)
  print("Emotion detected: ", custom)

# x_train, y_train, x_test, y_test = split_data()
create_model()
cnn()
test_cnn()