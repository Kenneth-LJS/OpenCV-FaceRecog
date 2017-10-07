# ==================== Remove this section if you can import cv2 without errors
import sys
sys.path.append('/usr/local/libpython3.6/site-packages') # for cv2
# ====================

import cv2
import numpy as np

faceThreshold = 65 # Lower = stricter accuracy, Higher = less strict

trainingFaces = []
trainingLabels = []

def findBiggestFace(faces):
	bestX = -1
	bestY = -1
	bestWidth = -1
	bestHeight = -1
	bestSize = 0
	for x, y, width, height in faces:
		if width * height > bestSize:
			bestSize = width * height
			bestX = x
			bestY = y
			bestWidth = width
			bestHeight = height
	if bestSize == 0:
		return None
	return (bestX, bestY, bestWidth, bestHeight)

def extractFace(frame, facePos):
	x, y, width, height = facePos
	return np.array(frame, 'uint8')[x:x+width,y:y+height]

vc = cv2.VideoCapture(0) # change index to get different camera
cc = cv2.CascadeClassifier("face_detector.xml")

recognizerTrained = False
recognizer = cv2.face.LBPHFaceRecognizer_create()

cv2.namedWindow('preview')

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

while rval:
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = cc.detectMultiScale(
			grayFrame,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE
		)
	
	# detect all faces
	for x, y, width, height in faces:
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

	# print biggest face in blue
	biggestFace = findBiggestFace(faces)
	if biggestFace != None:
		x, y, width, height = biggestFace
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

	# label faces
	if recognizerTrained:
		for x, y, width, height in faces:
			label, score = recognizer.predict(grayFrame[y:y+height,x:x+width])
			if (score <= faceThreshold):
				#cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
				cv2.putText(frame, str(label), (x + 5, y + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
				continue

	cv2.imshow('preview', frame)
	key = cv2.waitKey(20)

	if key == 27:
		# ESC: exit
		break
	elif key >= 48 and key <= 57:
		if (biggestFace != None):
			# 0-9: add training data on biggest face with specified label 0 to 9
			number = key - 48
			trainingFaces.append(extractFace(grayFrame, biggestFace))
			trainingLabels.append(number)
			print('Added new training data with label ' + str(number))
	elif key == 110:
		# N: new model
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizerTrained = False
		print('New model created')
	elif key == 116:
		# T: train model from training data
		if len(trainingFaces) == 0 or len(trainingLabels) == 0:
			print('No data in training data.')
		else:
			print('Training model.')
			if not recognizerTrained:
				recognizer.train(trainingFaces, np.array(trainingLabels))
				print('Model trained.')
			else:
				recognizer.update(trainingFaces, np.array(trainingLabels))
				print('Model updated.')
			recognizerTrained = True
			trainingFaces = []
			trainingLabels = []
	elif key == 115:
		# S: save model
		if recognizerTrained:
			recognizer.write('trained.yml')
			print('Model saved')
		else:
			print('Model is not trained. Unable to save.')
	elif key == 108:
		# L: load model
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizer.read('trained.yml')
		recognizerTrained = True
		print('Model loaded')
	#elif key != -1:
	#	# Read keypresses to add additional keypress functionality
	#	print('key: ' + str(key));

	rval, frame = vc.read()

cv2.destroyWindow("preview")