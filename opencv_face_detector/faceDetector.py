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

    # mask = cv.resize(mask, (340, 340), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, h, w) in faces:
        # print(h, w)
        center = int(w/2)+x, int(h/2)+y
        # cv.circle(frame, center, 5, (0, 0, 0), 2)

        # print(w+x, width/5, h, height)
        # center = (int(w/2)+x, int(h/2)+y)
        # cv.circle(frame, (center[0] - int(w/5),
        #           center[1]), 5, (255, 255, 0), 3)
        # cv.putText(frame, f'{center[0] - int(w/5)}',
        #            (center[0] - int(w/5), 30), fonts, 0.7, (0, 255, 0), 2)

        offSet = int(width/3.5)
        cv.line(frame, center, (center[0] -
                offSet, center[1]), (0, 0, 0), 3)

        cv.line(frame, center,
                (center[0] + offSet, center[1]), (0, 255, 255), 3)

        cv.putText(frame, f'{center[0] - offSet}',
                   (center[0] - offSet, center[1]), fonts, 0.7, (0, 255, 0), 2)

        cv.putText(frame, f'{center[0] + offSet}',
                   (center[0] + offSet, center[1]), fonts, 0.7, (255, 255, 0), 2)

        # cv.circle(frame,  (center[0] + int(w/5), 60), 5, (0, 255, 255), 3)
        # cv.circle(frame, center, 5, (0, 255, 0), 3)

        # if (x > 180 and (x+w) < width-180):

        # print(f'{(y+h)<height-180}, y = {height-(y+h) }')
        # cv.circle(frame, (40, y+h), 5, (0, 0, 255), 3)

        # region = frame[center[1]-180: center[1] +
        #                180, center[0]-180: center[0]+180]

        # print(region.shape)
        # out = cv.bitwise_and(region, mask)
        # cv.imshow('out', out)

        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #     cv.imshow('region', region)

    cv.imshow('image', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows
camera.release()
