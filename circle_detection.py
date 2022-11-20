import sys, os
import cv2 as cv
import numpy as np

#pylint: disable=no-member

CENTER = (304, 236)
IMG_COLOR = None
IMG_GRAY = None

def open_image(file_name):
    # Load an color image in grayscale
    global IMG_COLOR
    IMG_COLOR = cv.imread(cv.samples.findFile(file_name), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if IMG_COLOR is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + file_name + '] \n')
        sys.exit()
    IMG_GRAY = cv.cvtColor(IMG_COLOR, cv.COLOR_BGR2GRAY)
    IMG_GRAY = cv.medianBlur(IMG_GRAY, 5)
    return IMG_GRAY

def mark_center_of_holder(img):
    global IMG_COLOR
    
    # #circle center
    # cv.circle(IMG_COLOR, CENTER, 1, (0, 100, 100), 3)
    # #circle outline
    # radius = 115
    # cv.circle(IMG_COLOR, CENTER, radius, (255, 0, 255), 3)
    holder = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100,
                               param1=120, param2=30,
                               minRadius=110, maxRadius=150)

    if holder is not None:
        holder = np.uint16(np.around(holder))
        for i in holder[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(IMG_COLOR, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            cv.putText(IMG_COLOR, f"Coordinates of holder: {center[0]} , {center[1]} Radius: {i[2]}", (10, 400),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def detect_inner_circle(img):
    global IMG_COLOR
    #Detect the inner circle
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100,
                               param1=120, param2=30,
                               minRadius=40, maxRadius=100)

    # Mark the center of the inner circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(IMG_COLOR, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            cv.putText(IMG_COLOR, f"Coordinates of anode: {center[0]} , {center[1]}. Radius: {i[2]}", (10, 440),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def main():

    default_file = 'trial_anode/2022_11_17_14h_32m_53s.jpg'
    #default_file = sys.argv[1] if len(sys.argv) > 1  else default_file
    for file in os.listdir("trial_anode"):
        # join the path and filename
        path = os.path.join("trial_anode", file)
        img_gray = open_image(path)
        mark_center_of_holder(img_gray)
        #mark_center_of_holder(img_gray)
        
        #img_circle_color = detect_inner_circle(img_color)
        detect_inner_circle(img_gray)

        cv.imshow("detected circles", IMG_COLOR)
        cv.waitKey(5000)

    return 0
if __name__ == "__main__":
    main()