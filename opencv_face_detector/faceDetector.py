import cv2 as cv
import numpy as np

faceDetector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv.VideoCapture(0)
fonts = cv.FONT_HERSHEY_COMPLEX
mask = cv.imread('masks/mask 2.png')
cv.imshow('mask', mask)
print('mask ', mask.shape)

while True:
    ret, frame = camera.read()
    height, width, ch = frame.shape

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, h, w) in faces:
        center = int(w/2)+x, int(h/2)+y

        offSet = int(width/3.7)

        if (center[0]-offSet > 0) and (center[0]+offSet < width) and (center[1]-offSet > 0 and center[1] + offSet < height):
            ROI = frame[center[1]-offSet: center[1] +
                        offSet, center[0]-offSet: center[0]+offSet]
            cv.imshow("ROI", ROI)
            # print()
            height, width, _ = ROI.shape
            mask = cv.resize(mask, (height, width),
                             interpolation=cv.INTER_AREA)
            out = cv.bitwise_and(ROI, mask)
            # mask = cv.resize(mask, (340, 340), interpolation=cv.INTER_AREA)
            out[np.where((mask == [0, 0, 0]).all(axis=2))] = [155, 0, 255]
            cv.imshow('out', out)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #     cv.imshow('region', region)

    cv.imshow('image', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows
camera.release()
