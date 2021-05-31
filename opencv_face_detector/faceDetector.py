import cv2 as cv
import numpy as np

faceDetector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv.imshow('image', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows
camera.release()
