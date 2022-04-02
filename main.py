import cv2

# Initiate video capture using webcam
print("Starting up the webcam...")
camera = cv2.VideoCapture(0)

# loop over the frames from the video stream while camera is active
while camera.isOpened():
	# current frame
	s, frame = camera.read()

	# if no input from camera, skip to next iteration
	if not s:
		continue

	# downscale to reduce size of image, achieve faster computation
	scale_percent = 70 # percent of original size
	width = int(frame.shape[1] * scale_percent / 100)
	height = int(frame.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image using INTER_AREA for interpolation
	frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", frame)

	# print("shape of frame: ", frame.shape)
	# shape of frame:  (720, 1280, 3)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# print("shape of gray: ", gray.shape)
	# shape of gray:  (720, 1280)
        
    # show the gray frame
	cv2.imshow("Gray Frame", gray)
 
	# if the `q` key was pressed, break from the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# Close all windows, close camera
cv2.destroyAllWindows()
camera.release()