import numpy as np
import os

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr


# restructuring classifications for TESS database
tess_emotions = {'neutral': '01', 'happy': '03', 'sad': '04', 'angry': '05', 'fear': '06', 'disgust': '07', 'ps': '08'}

# feature data
rms = []
zcr = []
mfcc = []
emotions = []

total_length = 173056 # desired frame length for all of the audio samples.

folder_path = 'datasets/' 

# get both TESS and RAVDESS data sets
for subdir, dirs, files in os.walk(folder_path):
    for file in [f for f in files if not f[0] == '.']: # to ignore hidden files
        file_path = os.path.join(subdir, file)
        # normalize sound from file
        sound = effects.normalize(AudioSegment.from_file(file_path), headroom = 0)
        x = np.array(sound.get_array_of_samples(), dtype = 'float32')
        # trim silences from ends of sound
        x, index = librosa.effects.trim(x, top_db=30)
        # pad sounds so they all have same length
        x = np.pad(x, (0, total_length - len(x)), 'constant')
        # reduce noise
        _, sr = librosa.load(path = file_path, sr = None) 
        x = nr.reduce_noise(x, sr = sr) 
        
        # features extraction 
        # root-mean-square (RMS) value for each frame
        rms_x = librosa.feature.rms(y = x)   
        # zero-crossing rate of an audio time series
        zcr_x = librosa.feature.zero_crossing_rate(x)   
        # Mel-frequency cepstral coefficients
        mfcc_x = librosa.feature.mfcc(y = x, sr = sr, n_mfcc = 13, hop_length = 512)
        
        # emotion extraction
        assigned = False
        for emotion in tess_emotions.keys():
            # TESS database validation
            if emotion in file:
                name = tess_emotions[emotion]
                assigned = True
                break
        
        # RAVDESS database validation
        if not assigned:
            name = file[6:8]
        

        rms.append(rms_x)
        zcr.append(zcr_x)
        mfcc.append(mfcc_x)
        # shifting classification to start from 0
        emotions.append(int(name) - 1)

rms = np.swapaxes(np.asarray(rms).astype('float32'), 1, 2)
zcr = np.swapaxes(np.asarray(zcr).astype('float32'), 1, 2)
mfcc = np.swapaxes(np.asarray(mfcc).astype('float32'), 1, 2)

# features are set as input data
x = np.concatenate((zcr, rms, mfcc), axis=2)
print(x.shape)

# emotions for classificiation
y = np.expand_dims(np.asarray(emotions).astype('int8'), axis = 1)
print(y.shape)

# save data
np.save('data/features.npy', x)
np.save('data/emotions.npy', y)