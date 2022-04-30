import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np

num_buckets = 2
batch_size = 256
steps_per_epoch = 112
epochs = 11
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
buckets = ['negative', 'nonnegative'] # 0: negative, 1: nonnegative
map_to_bucket = {'0':'0', '1':'0', '2':'0', 
                    '3':'1', '4':'0', '5':'0', '6':'1'}

def split_data():
  # Import fer2013.csv
  with open("../data/fer2013.csv") as file:
      data = file.readlines()
  lines = np.array(data)
  x_train, y_train, x_test, y_test = [], [], [], []

  # Split dataset into training and test sets
  for i in range(1,lines.size):
      emotion, img, usage = lines[i].split(",")
      val = img.split(" ")
      pixels = np.array(val, 'float32')
      emotion = keras.utils.np_utils.to_categorical(map_to_bucket[emotion], num_buckets)
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
  
  # Use increasing number of filters in Conv2D
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
  pred = Dense(2, activation='softmax')(dense)
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
  model.save('../models/model.h5')

def test_cnn():
  model = load_model('../models/omar178.h5')
  # x_train, y_train, x_test, y_test = split_data()
  # model.evaluate(x_test, y_test)
  img = image.load_img("../data/jinkim.png", color_mode = "grayscale", target_size=(64, 64))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis = 0)
  x /= 255
  custom = model.predict(x)
  print("Emotion detected: ", custom)

# cnn()
test_cnn()