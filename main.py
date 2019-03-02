import cv2

vidStream = cv2.VideoCapture(0)
faceClass = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = vidStream.read()
	grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = faceClass.detectMultiScale(grayFrame,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	
	cv2.imshow("Colour Frame",frame)

	#Press 'q' to stop
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

vidStream.release()
cv2.destroyAllWindows()