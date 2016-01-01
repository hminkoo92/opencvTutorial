# 처음 깔았을 때 해야 할 일 
* 설치방법(설치 사이트: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows)

    1. 사이트 접속 후, Installing OpenCV from prebuilt binaries 부분 따라해서 설치하기. 

* 예제 코드 잘 돌아가는지 확인. (예제 코드 참고 사이트 : http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html)

* 예제 코드
```python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame 
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
        
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

* 라이브러리가 없는 경우에는 추가(pip 사용)
    
    1. pip 사용법 
        1. WindowPowerShell(Ctrl + Shift + T) 실행
        1. pip install LibraryName 입력
        1. import <library name> 을 통하여 설치 여부 확인.


# Face Recognition
 
* 참고 사이트:
    1.  http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    1.  http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

* 예제 코드
```python
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('sachin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

