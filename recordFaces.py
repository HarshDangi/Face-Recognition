import cv2
import numpy as np

dataPath = "./faceData/"
personName = input("Enter person name : ")

vidStream = cv2.VideoCapture(0)
faceClass = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
faceData = []

while True:
	ret,frame = vidStream.read()

	if ret == False:
		continue

	grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = faceClass.detectMultiScale(grayFrame,1.3,5)

	#getting largest face
	face = sorted(faces,key = lambda f:f[2]*f[3])[-1:]

	for (x,y,w,h) in face:
		#printing rectangle around selected face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		
		offset = 10
		finalFace = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		finalFace = cv2.resize(finalFace,(100,100))
		cv2.imshow("Cropped",finalFace)


		if skip%10==0:
			faceData.append(finalFace)
			print("pics clicked - ",int(skip/10)+1)

		skip += 1

	cv2.imshow("Colour Frame",frame)

	#Press 'q' to stop
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

#saving faceData
faceData = np.asarray(faceData)
if faceData.shape[0] != 0:
	faceData = faceData.reshape((faceData.shape[0],-1))
	np.save(dataPath+personName+".npy",faceData)
	print("Data saved successfully")
else:
	print("No face Detected")

vidStream.release()
cv2.destroyAllWindows()