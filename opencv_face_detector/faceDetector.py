import cv2 as cv
import numpy as np

faceDetector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, h, w) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv.imshow('image', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows
camera.release()
