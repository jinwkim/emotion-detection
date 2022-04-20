from json import load
from charset_normalizer import detect
import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import socket
import cv2
import pickle
import struct
import os

import librosa
from pydub import AudioSegment, effects
import noisereduce as nr
from array import array
import wave
import pyaudio

from notification import displayNotification
import asyncio

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '127.0.0.1' 
port = 1234
print("Socket Created Successfully")
client_socket.connect((host_ip,port))
data = b""
payload_size = struct.calcsize("Q")
print("Socket Accepted")

# Initialize for emotion detection from face
counter = 0
model = load_model('models/omar178.h5') # load pre-trained model
speech_model = load_model('models/speech_emotion.h5')
speech_emotions = {
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
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
neg_count_face, pos_count_face = 0,0
neg_count_speech, pos_count_speech = 0,0
negative_emotions = {'angry','disgust','fear','sad','surprise'}
detected_emotion = 'neutral'

max_emotion = "N/A"

# Initialize variables
RATE = 24414
CHUNK = 512
RECORD_SECONDS = 7.1

FORMAT = pyaudio.paInt32
CHANNELS = 1
WAVE_OUTPUT_FILE = "data/output.wav"

# Open an input channel
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


def preprocess(file_path, frame_length = 2048, hop_length = 512):
    '''
    A process to an audio .wav file before execcuting a prediction.
      Arguments:
      - file_path - The system path to the audio file.
      - frame_length - Length of the frame over which to compute the speech features. default: 2048
      - hop_length - Number of samples to advance for each frame. default: 512

      Return:
        'X_3D' variable, containing a shape of: (batch, timesteps, feature) for a single file (batch = 1).
    ''' 
    rawsound = file_path; sr = RATE 
    # Normalize to 5 dBFS 
    normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32') 
    # Noise reduction                  
    final_x = nr.reduce_noise(normal_x, sr=sr)
        
        
    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # Energy - Root Mean Square
    f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length,center=True).T # ZCR
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T # MFCC   
    X = np.concatenate((f1, f2, f3), axis = 1)
    
    X_3D = np.expand_dims(X, axis=0)
    
    return X_3D

# Initialize a non-silent signals array to state "True" in the first 'while' iteration.
speech_data = array('h', np.random.randint(size = 512, low = 0, high = 500))
speech_frames = []
timesteps = int(RATE / CHUNK * RECORD_SECONDS) # => 339
print(timesteps)

while True:
	try:
		print("* recording...")

		speech_data = array('l', stream.read(CHUNK, exception_on_overflow = False)) 
		speech_frames.append(speech_data)
		speech_frames.append(speech_data)
		speech_frames.append(speech_data)
		print('frames length', len(speech_frames))

		while len(data) < payload_size:
			packet = client_socket.recv(2160) 
			if not packet: 
				break
			data += packet
		packed_msg_size = data[:payload_size]
		data = data[payload_size:]
		msg_size = struct.unpack("Q",packed_msg_size)[0]
		
		while len(data) < msg_size:
			data += client_socket.recv(2160)
		frame_data = data[:msg_size]
		data  = data[msg_size:]
		frame = pickle.loads(frame_data)

		# Detect emotion from incoming video stream
		resized = cv2.resize(frame, (64,64))
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

		counter += 1

		if counter % 5 == 0: # 30 fps
			img = gray
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis = 0)
			x /= 255
			pred_vals = model.predict(x)[0] # [[0.09034569, 0.04079238, 0.13130878, 0.06450415, 0.44670576, 0.08373094, 0.14261228]]
			detected_emotion = emotions[np.argmax(pred_vals)]
			# s = "'angry' {0}, 'disgust' {1}, 'fear' {2}, 'happy' {3}, 'sad' {4}, 'surprise' {5}, 'neutral' {6}"
			# print(s.format(*pred_vals))
			# print("Detected emotion: ", emotions[np.argmax(pred_vals)])
			if detected_emotion in negative_emotions:
				neg_count_face += 1
			else:
				pos_count_face += 1


		if (len(speech_frames) - 1) % timesteps == 0: 
			print('len(speech_frames): ',len(speech_frames))
			print("timesteps: ", timesteps)
			print('counter', counter)
			speech_frames = speech_frames[:-1]
			# x = preprocess(WAVE_OUTPUT_FILE) # 'output.wav' file preprocessing.
			audio_segment = AudioSegment(
				b''.join(speech_frames), 
				frame_rate=RATE,
				sample_width=p.get_sample_size(FORMAT), 
				channels=CHANNELS
			)
			x = preprocess(audio_segment)
			# print('here2')
			# print(x.shape)
			# Model's prediction => an 8 emotion probabilities array.
			predictions = speech_model.predict(x, use_multiprocessing=True)
			pred_list = list(predictions)
			pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0) # Get rid of 'array' & 'dtype' statments.
			
			
			max_emo = np.argmax(predictions)
			emotion_level = map_to_bucket[str(max_emo)]
			print("Emotion level from speech: ", emotion_level)
			max_emotion = speech_emotions.get(max_emo,-1)
			print('max emotion:', speech_emotions.get(max_emo,-1))

			if max_emotion in negative_emotions:
				neg_count_speech += 1
			else:
				pos_count_speech += 1

			speech_frames = []

		# Show the incoming video from transmitter.py
		cv2.putText(frame, "Detected patient's emotion: "+detected_emotion, (20,30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
		cv2.putText(frame, "Emotion from speech: "+max_emotion, (20,70),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
		cv2.imshow("RECEIVING VIDEO",frame)

		# Show notification if negative emotions persist for 5 seconds (150/30)
		if counter % 338 == 0:
			face_ratio = 1.0 * neg_count_face / (neg_count_face + pos_count_face)
			speech_ratio = 1.0 * neg_count_speech / (neg_count_speech + pos_count_speech)
			print("face_ratio: {}, speech_ratio: {}".format(face_ratio, speech_ratio))
			# if neg_count_face >= pos_count_face or neg_count_speech >= pos_count_speech:
			if face_ratio*0.7 + speech_ratio*0.3 >= 0.45:
				displayNotification(message="Your patient may be experiencing negative emotions. Please attend to them right away", 
									title="Patient Needs Your Attention")
			neg_count_face, neg_count_speech, pos_count_face, pos_count_speech = 0,0,0,0

		# If "q" is pressed, break
		key = cv2.waitKey(1) & 0xFF
		if key  == ord('q'):
			break
	except Exception as err:
		print(err)
		print("error in receiver.py")
		client_socket.close()
		cv2.destroyAllWindows()
		stream.stop_stream()
		stream.close()
		p.terminate()
		# wf.close()
stream.stop_stream()
stream.close()
p.terminate()
# wf.close()
client_socket.close()
cv2.destroyAllWindows()