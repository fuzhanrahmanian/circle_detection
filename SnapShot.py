import time
import os
import numpy as np
import cv2 as cv

CAM_PORT_BTM = 1
CAM_PORT_TOP = 2

choose_camera = input("Choose Camera( [1]-->Bottom Cam; [2]-->Top Cam ): ")
if choose_camera == '1':
    alpha = 1.0 # Simple contrast control [1.0-3.0]
    beta = 5    # Simple brightness control [0-100]
    cap = cv.VideoCapture(CAM_PORT_BTM,cv.CAP_DSHOW)
elif choose_camera == '2':
    alpha = 2.0 # Simple contrast control [1.0-3.0]
    beta = 40    # Simple brightness control [0-100]
    cap = cv.VideoCapture(CAM_PORT_TOP,cv.CAP_DSHOW)
# cv.namedWindow("Camera Calibration")

if cap.isOpened():
    ret, frame = cap.read() # Try to get the first frame
else:
    ret = False
    print("Cannot open camera")
    exit()
mark = False
while ret is True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if mark == True:
        (h, w) = frame.shape[:2]
        center = (w//2, h//2) # where w//2, h//2 are the required frame/image centeroid's XYcoordinates.
        radius = 20
        cv.circle(frame, center, radius, (0, 0, 255), 1)
        cv.line(img=frame, pt1=(center[0]-2*radius, center[1]), pt2=(center[0]+2*radius, center[1]), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
        cv.line(img=frame, pt1=(center[0], center[1]-2*radius), pt2=(center[0], center[1]+2*radius), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
    cv.imshow("Camera Calibration -- Press [Q] to exit", frame)
    
    # if frame is read correctly ret is True
    key = cv.waitKey(20)
    if key == ord('q'):
        break
    elif key == ord('s'):
        time_stamp = time.strftime("%Y_%m_%d_%Hh_%Mm_%Ss", time.localtime())
        dir_name = os.path.join(os.path.dirname(__file__), 'Alignments', time_stamp[:10])
        filename = f"[Camera{choose_camera}_Calib]_{time_stamp}.jpg"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        cv.imwrite(filename, frame)
        print(f"Image has been saved: {dir_name}")
    elif key == ord('o'):
        mark = True
    elif key == ord('c'):
        mark = False
    # Our operations on the frame come here
    # Display the resulting frame
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()