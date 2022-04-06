# https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob 
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

emotions = ['calm', 'sad', 'fearful', 'calm', 'fearful', 'happy', 'calm', 'fearful', 'angry', 'happy']

data, sampling_rate = librosa.load('../data/output10.wav')

# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(data, sr=sampling_rate)
# plt.show()

# data, sampling_rate = librosa.load('../data/output10.wav')
model = load_model('../models/Emotion_Voice_Detection_Model.h5')
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

X, sample_rate = librosa.load('../data/output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
twodim= np.expand_dims(livedf2, axis=2)
livepreds = model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)
livepreds1=livepreds.argmax(axis=1)
liveabc = livepreds1.astype(int).flatten()
detected_emotion = emotions[liveabc[0]]
