import keras
from keras.preprocessing import image
from keras.models import load_model

import numpy as np
import cv2

# Load existing model
# model = load_model('models/omar178.h5')
# emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

model = load_model('../models/face-emotion.h5')
emotions = ['negative', 'nonnegative']

# Initiate video capture using webcam
print("Starting up the webcam...")
camera = cv2.VideoCapture(0)

# loop over the frames from the video stream while camera is active
# fps = 30
counter = 0
detected_emotion = "None"
while camera.isOpened():
	# current frame - numpy.ndarray
	s, frame = camera.read()

	# if no input from camera, skip to next iteration
	if not s:
		continue
	
	# cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
	cv2.putText(frame, detected_emotion, (20,30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", frame)

	# downscale to reduce size of image, achieve faster computation
	# scale_percent = 70 # percent of original size
	# width = int(frame.shape[1] * scale_percent / 100)
	# height = int(frame.shape[0] * scale_percent / 100)
	# dim = (width, height)
	# # resize image using INTER_AREA for interpolation
	# frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	resized = cv2.resize(frame, (48,48))
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

	if counter % 60 == 0: # 30 fps
		img = gray
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis = 0)
		x /= 255
		pred_vals = model.predict(x)[0] # [[0.09034569, 0.04079238, 0.13130878, 0.06450415, 0.44670576, 0.08373094, 0.14261228]]
		print("pred_vals: ", pred_vals)
		detected_emotion = emotions[np.argmax(pred_vals)]
		# s = "'angry' {0}, 'disgust' {1}, 'fear' {2}, 'happy' {3}, 'sad' {4}, 'surprise' {5}, 'neutral' {6}"
		# print(s.format(*pred_vals))
		print("Detected emotion: ", emotions[np.argmax(pred_vals)])
	counter += 1

	# if the `q` key was pressed, break from the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# Close all windows, close camera
cv2.destroyAllWindows()
camera.release()