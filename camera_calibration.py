#https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html

import os
import numpy as np
import cv2 as cv
import glob
import time

os.chdir(os.path.dirname(__file__))
savedir="camera_data/"

# termination criteria

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)

#add 2.5 to account for 2.5 cm per square in grid
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)*2.5

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
markimgs = []

images = glob.glob('calibration_images/*.jpg')

win_name="Verify"
cv.namedWindow(win_name, cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty(win_name,cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)

print("getting images")
for fname in images:
    img = cv.imread(fname)
    print(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        markimgs.append(img)
        cv.imshow(win_name, img)
        cv.waitKey(500)

    img1=img
    
cv.destroyAllWindows()

print(">==> Starting calibration")
ret, cam_mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#print(ret)
print("Camera Matrix")
print(cam_mtx)
np.save(savedir+'cam_mtx.npy', cam_mtx)

print("Distortion Coeff")
print(dist)
np.save(savedir+'dist.npy', dist)

print("r vecs")
print(rvecs[2])

print("t Vecs")
print(tvecs[2])



print(">==> Calibration ended")

for img in markimgs:
    h,  w = img.shape[:2]
    newcam_mtx, roi=cv.getOptimalNewCameraMatrix(cam_mtx, dist, (w,h), 1, (w,h))
    undst = cv.undistort(img, cam_mtx, dist, None, newcam_mtx)
    cv.imshow('origin', img)
    cv.imshow('undistored', undst)
    cmd = cv.waitKey(0)
    if cmd == ord('q'):
        break
    elif cmd == ord('d'):
        continue

# h,  w = img1.shape[:2]
# print("Image Width, Height")
# print(w, h)
# #if using Alpha 0, so we discard the black pixels from the distortion.  this helps make the entire region of interest is the full dimensions of the image (after undistort)
# #if using Alpha 1, we retain the black pixels, and obtain the region of interest as the valid pixels for the matrix.
# #i will use Apha 1, so that I don't have to run undistort.. and can just calculate my real world x,y
# newcam_mtx, roi=cv.getOptimalNewCameraMatrix(cam_mtx, dist, (w,h), 1, (w,h))

# print("Region of Interest")
# print(roi)
# np.save(savedir+'roi.npy', roi)

# print("New Camera Matrix")
# #print(newcam_mtx)
# np.save(savedir+'newcam_mtx.npy', newcam_mtx)
# print(np.load(savedir+'newcam_mtx.npy'))

# inverse = np.linalg.inv(newcam_mtx)
# print("Inverse New Camera Matrix")
# print(inverse)


# # undistort

# undst = cv.undistort(img1, cam_mtx, dist, None, newcam_mtx)



# # crop the image
# #x, y, w, h = roi
# #dst = dst[y:y+h, x:x+w]
# #cv.circle(dst,(308,160),5,(0,255,0),2)
# cv.imshow('origin', img1)
# cv.imshow('img1', undst)
# cv.waitKey(0) 
# # cv.waitKey(5000)      
# # cv.destroyAllWindows()
# # cv.imshow('img1', undst)
# # cv.waitKey(5000)      
# # cv.destroyAllWindows()