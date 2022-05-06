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

import dlib
from datetime import datetime

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
# load pre-trained model
model = load_model('models/face-emotion.h5')  # loss: 1.0510 - acc: 0.6041 - val_loss: 1.0856 - val_acc: 0.5904
speech_model = load_model('models/speech_emotion.h5')
# map classification number to emotion name
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

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

neg_count_face, pos_count_face = 0, 0
neg_count_speech, pos_count_speech = 0, 0
negative_emotions = {'angry','disgust','fear','sad', 'fearful', 'surprise', 'surprised'}

emotion_face = 'neutral'
emotion_speech = 'N/A'

# boolean to see if face detected
face_detected = False

RATE = 24414
CHUNK = 512
RECORD_SECONDS = 7.1
FORMAT = pyaudio.paInt32
CHANNELS = 1 # Macbook supports 1 channel

# Open an input channel
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

file_name = datetime.now()
file_name = "log/"+str(file_name)+".csv"
file = open(file_name,"x")
file.close()

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

# Initialize a non-silent signals array to state "True" in the first 'while' iteration.
speech_data = array('h', np.random.randint(size = 512, low = 0, high = 500))
speech_frames = []
timesteps = int(RATE / CHUNK * RECORD_SECONDS) # => 339
print(timesteps)

while True:
	try:
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
		resized = cv2.resize(frame, (48,48))
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

		print("* recording speech...")

		speech_data = array('l', stream.read(CHUNK, exception_on_overflow = False)) 
		# adding three frames at a time to make speech recognition faster
		speech_frames.append(speech_data)
		speech_frames.append(speech_data)
		speech_frames.append(speech_data)
		print('frames length', len(speech_frames))

		counter += 1

		if counter % 5 == 0: # 30 fps
			# If no face detected, skip to next iteration - save computing power
			gray_dlib = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			detector = dlib.get_frontal_face_detector()
			faces_detected = detector(gray_dlib, 0)
			if len(faces_detected) == 0:
				print("No faces found, defaulting to neutral")
				emotion_face = 'neutral' # if face not detected, default to 'neutral'
			else:
				img = gray
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis = 0)
				x /= 255
				pred_vals = model.predict(x)[0] # [[0.09034569, 0.04079238, 0.13130878, 0.06450415, 0.44670576, 0.08373094, 0.14261228]]
				emotion_face = emotions[np.argmax(pred_vals)]
				# s = "'angry' {0}, 'disgust' {1}, 'fear' {2}, 'happy' {3}, 'sad' {4}, 'surprise' {5}, 'neutral' {6}"
				# print(s.format(*pred_vals))
				# print("Detected emotion: ", emotions[np.argmax(pred_vals)])

			print("detected facial emotion: ", emotion_face)

			if emotion_face in negative_emotions:
				neg_count_face += 1
			else:
				pos_count_face += 1


		if (len(speech_frames) - 1) % timesteps == 0: 
			speech_frames = speech_frames[:-1]
			audio_segment = AudioSegment(
				b''.join(speech_frames), 
				frame_rate=RATE,
				sample_width=p.get_sample_size(FORMAT), 
				channels=CHANNELS
			)
			x = preprocess(audio_segment)
			preds = list(speech_model.predict(x, use_multiprocessing=True))
			emotion_speech = speech_emotions[np.argmax(preds)]
			print('Emotion from speech:', emotion_speech)

			speech_frames = []

		# Show the incoming video from transmitter.py
		# cv2.putText(frame, "Emotion from facial expressions: " + emotion_face, (20,30),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
		# cv2.putText(frame, "Emotion from speech: " + emotion_speech, (20,70),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
		cv2.imshow("RECEIVING VIDEO",frame)

		# Show notification if negative emotions persist for 5 seconds (150/30)
		if counter % 338 == 0:
			face_ratio = 1.0 * neg_count_face / (neg_count_face + pos_count_face)
			
			print("face_ratio: {}, emotion_speech: {}".format(face_ratio, emotion_speech))

			if face_ratio >= 0.5 or emotion_speech in negative_emotions:
				displayNotification(message="Your patient may be experiencing negative emotions. Please attend to them right away", 
									title="Patient Needs Your Attention")
				file = open(file_name, "a")
				now = datetime.now()
				file.write(str(now)+",emotion_face,"+emotion_face+",emotion_speech,"+emotion_speech+",alert\n")
				file.close()
			neg_count_face, neg_count_speech, pos_count_face, pos_count_speech = 0,0,0,0
			
		# If "q" is pressed, break
		key = cv2.waitKey(1) & 0xFF
		if key  == ord('q'):
			break
	except Exception as err:
		print(err)
		print("error in receiver.py")
		file.close()
		client_socket.close()
		cv2.destroyAllWindows()
		stream.stop_stream()
		stream.close()
		p.terminate()
		# wf.close()

stream.stop_stream()
stream.close()
file.close()
p.terminate()
client_socket.close()
cv2.destroyAllWindows()