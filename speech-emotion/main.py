import numpy as np

import librosa
from pydub import AudioSegment, effects
import noisereduce as nr

import keras
from keras.models import load_model

import pyaudio
import wave
from array import array

# Initialize variables
RATE = 24414
CHUNK = 512
RECORD_SECONDS = 7.1

FORMAT = pyaudio.paInt32
CHANNELS = 1
emotions = {
    0 : 'neutral',
    1 : 'calm',
    2 : 'happy',
    3 : 'sad',
    4 : 'angry',
    5 : 'fearful',
    6 : 'disgust',
    7 : 'suprised'   
}
map_to_bucket = {'0':'0', '1':'0', '2':'0', 
                    '3':'1', '4':'1', '5':'1', '6':'1'}

model = load_model('../models/speech_emotion.h5')

def preprocess(audio_segment):
	# normalize sound
	sound = effects.normalize(audio_segment, headroom = 5.0)
	x = np.array(sound.get_array_of_samples(), dtype = 'float32')
	# reduce noise
	x = nr.reduce_noise(x, sr = RATE)

	# features extraction 
    # root-mean-square (RMS) value for each frame
	rms_x = librosa.feature.rms(y = x, pad_mode = 'reflect').T   
	# zero-crossing rate of an audio time series
	zcr_x = librosa.feature.zero_crossing_rate(x).T   
	# Mel-frequency cepstral coefficients
	mfcc_x = librosa.feature.mfcc(y = x, sr = RATE, n_mfcc = 13, hop_length = 512).T
	x = np.concatenate((rms_x, zcr_x, mfcc_x), axis = 1)

	return np.expand_dims(x, axis = 0) # format: (batch, timesteps, feature)

# Emotions list is created for a readable form of the model prediction.

# Open an input channel
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# initialize sound signals
data = array('h', np.random.randint(size = 512, low = 0, high = 500))
timesteps = int(RATE / CHUNK * RECORD_SECONDS) # => 339

while True:
    print("recording new chunk")
    frames = [] 
    data = np.nan

    for i in range(0, timesteps):
        data = array('l', stream.read(CHUNK, exception_on_overflow = False)) 
        frames.append(data)

    print('frames collected')
    audio_segment = AudioSegment(
				b''.join(frames), 
				frame_rate=RATE,
				sample_width=p.get_sample_size(FORMAT), 
				channels=CHANNELS
	)
    x = preprocess(audio_segment)
    preds = list(model.predict(x, use_multiprocessing=True))
    max_emotion = np.argmax(preds)
    emotion_speech = emotions[max_emotion]
    print('predicted emotion', emotion_speech)
    if map_to_bucket[str(max_emotion)] == '1':
        print('negative emotion detected')

    
stream.stop_stream()
stream.close()
p.terminate()

