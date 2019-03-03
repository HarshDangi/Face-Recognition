import numpy as np 
import cv2
import os

# KNN Algorithm

def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())

def knn(train,test,k=5):
	dist = []
	for i in range(train.shape[0]):
		ix = train[i,:-1]
		iy = train[i,-1]

		d = distance(test,ix)
		dist.append((d,iy))

	dk = sorted(dist,key=lambda x:x[0])
	labels = np.array(dk)[:,-1]

	newLables = np.unique(labels,return_counts=True)
	index = np.argmax(newLables[1])
	return newLables[0][index]

# identification algorithm

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataPath = "./faceData/"

faceData = []
labels = []

classID = 0
names = {}

for fx in os.listdir(dataPath):
	if fx.endswith('.npy'):
		#get faces
		dataItem = np.load(dataPath+fx)
		faceData.append(dataItem)

		names[classID] = fx[:-4]

		#create labels
		target = classID*np.ones((dataItem.shape[0],))
		classID += 1
		labels.append(target)

	faceDataset = np.concatenate(faceData,axis=0)
	faceLabels = np.concatenate(labels,axis=0).reshape((-1,1))

print(faceDataset.shape)
print(faceLabels.shape)

trainset = np.concatenate((faceDataset,faceLabels),axis=1)

print(trainset.shape)

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = faceCascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		#Get the face ROI
		offset = 10
		faceSection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		faceSection = cv2.resize(faceSection,(100,100))

		# get the prediction
		out = knn(trainset,faceSection.flatten())

		#Display on frame
		prediction = names[int(out)]
		cv2.putText(frame,prediction,(x,y-offset),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()