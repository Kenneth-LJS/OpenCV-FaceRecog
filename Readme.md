# Setup

1. Install Python 3.6
1. Install OpenCV 2.6 + opencv_contrib
	* https://github.com/opencv/opencv
	* https://github.com/opencv/opencv_contrib
	* https://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
1. Run script.py. A window should appear, using the default webcam. You can change the webcam being used by changing the 0 on the line 'vc = cv2.VideoCapture(0)'.
1. While focused on the webcam feed, various functionality can be triggered with their respective keypresses:
	* Esc:	Quits the application
	* N:	Create a new face recognition model
	* 0-9:	The biggest face on the webcam (outlined in blue) will be added to the training data with the label 0-9 (depending on the key pressed).
	* T:	Train the model based on the training data collected.
	* S:	Save the model to 'trained.yml'.
	* L:	Load the model from 'trained.yml'.

Other links:
* face_detector.xml: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
* Face/eye detectors: https://github.com/rsms/opencv-face-track-basics