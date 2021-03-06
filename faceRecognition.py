# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\Users\dakku\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture img 
    ret, img = cap.read()

    # Our operations on the frame here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check Face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    # Display the resulting img
    cv2.imshow('face',img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

