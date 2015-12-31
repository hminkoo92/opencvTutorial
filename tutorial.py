# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2
# 카메라를 쓸 준비를 한다...
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame / 읽어들인다. ret: 정상작동유무, frame :캡처한 정보를 담고있음
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture // release는 놓아준다.
cap.release()
cv2.destroyAllWindows()

