import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import socket
import cv2
import pickle
import struct
import os

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
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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
		resized = cv2.resize(frame, (64,64))
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
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
		counter += 1

		# Show the incoming video from transmitter.py
		cv2.putText(frame, "Detected patient's emotion: "+detected_emotion, (20,30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
		cv2.imshow("RECEIVING VIDEO",frame)

		# Show notification
		displayNotification(message="Your patient is here", title="Your Patient Needs Your Attention")

		# If "q" is pressed, break
		key = cv2.waitKey(1) & 0xFF
		if key  == ord('q'):
			break
	except:
		print("error in receiver.py")
		client_socket.close()
		cv2.destroyAllWindows()
client_socket.close()
cv2.destroyAllWindows()