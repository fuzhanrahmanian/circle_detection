import sys, os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#pylint: disable=no-member

CENTER = (304, 236)
IMG_COLOR = None
IMG_GRAY = None
DETECTED_CASE = []
DETECTED_ELECTRODE = []

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
    #IMG_GRAY = cv.GaussianBlur(IMG_GRAY, (1,1), 1)
    IMG_GRAY = cv.medianBlur(IMG_GRAY, 3)
    #IMG_GRAY = cv.adaptiveThreshold(IMG_GRAY,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3.5)
    #kernel = np.ones((2,2),np.uint8)
    IMG_GRAY = cv.erode(IMG_GRAY, (3,3), iterations=1)
    IMG_GRAY = cv.dilate(IMG_GRAY, (3,3), iterations=1)
    #IMG_GRAY = cv.blur(IMG_GRAY, (3, 3))
    return IMG_GRAY

def mark_center_of_holder(img):
    global IMG_COLOR
    
    # #circle center
    # cv.circle(IMG_COLOR, CENTER, 1, (0, 100, 100), 3)
    # #circle outline
    # radius = 115
    # cv.circle(IMG_COLOR, CENTER, radius, (255, 0, 255), 3)
    holder = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100,
                               param1=120, param2=20,
                               minRadius=115, maxRadius=125)

    if holder is not None:
        holder = np.uint16(np.around(holder))
        for i in holder[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(IMG_COLOR, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            DETECTED_CASE.append(radius)
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            cv.putText(IMG_COLOR, f"Coordinates of holder: {center[0]} , {center[1]} Radius: {i[2]}", (10, 400),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #print(f"Casing: Center_({center[0]} , {center[1]}), Radius_{i[2]}")

def mark_reference_center(img):
    global IMG_COLOR
    
    # #circle center
    # cv.circle(IMG_COLOR, CENTER, 1, (0, 100, 100), 3)
    # #circle outline
    # radius = 115
    # cv.circle(IMG_COLOR, CENTER, radius, (255, 0, 255), 3)
    holder = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100,
                               param1=120, param2=20,
                               minRadius=10, maxRadius=20)

    if holder is not None:
        holder = np.uint16(np.around(holder))
        for i in holder[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(IMG_COLOR, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            DETECTED_CASE.append(radius)
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            cv.putText(IMG_COLOR, f"Coordinates of Reference: {center[0]} , {center[1]} Radius: {i[2]}", (10, 400),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #print(f"Casing: Center_({center[0]} , {center[1]}), Radius_{i[2]}")

def detect_anode_circle(img):
    global IMG_COLOR
    #Detect the inner circle
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100,
                               param1=120, param2=25,
                               minRadius=85, maxRadius=96)

    # Mark the center of the inner circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(IMG_COLOR, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            DETECTED_ELECTRODE.append(radius)
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            cv.putText(IMG_COLOR, f"Coordinates of anode: {center[0]} , {center[1]}. Radius: {i[2]}", (10, 440),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #print(f"Anode: Center_({center[0]} , {center[1]}), Radius_{i[2]}")

def detect_cathode_circle(img):
    global IMG_COLOR
    #Detect the inner circle
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100,
                               param1=120, param2=22,
                               minRadius=86, maxRadius=92)

    # Mark the center of the inner circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(IMG_COLOR, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            DETECTED_ELECTRODE.append(radius)
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            cv.putText(IMG_COLOR, f"Coordinates of cathode: {center[0]} , {center[1]}. Radius: {i[2]}", (10, 440),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #print(f"Cathode: Center_({center[0]} , {center[1]}), Radius_{i[2]}")

def main():

    default_file = 'trial_anode/anode/2022_11_17_14h_32m_53s.jpg'
    #default_file = sys.argv[1] if len(sys.argv) > 1  else default_file
    electrode = input("Folder to test(anode/cathode): ")
    folder = os.path.join(r"trial_anode", electrode)
    group = []
    for file in os.listdir(folder):
        group.append(re.findall(r'\d+', file)[0])
        # join the path and filename
        path = os.path.join(folder, file)
        img_gray = open_image(path)
        #mark_center_of_holder(img_gray)
        mark_reference_center(img_gray)
        
        #img_circle_color = detect_inner_circle(img_color)
        if electrode == 'anode':
            detect_anode_circle(img_gray)
        elif electrode == 'cathode':
            detect_cathode_circle(img_gray)

        cv.imshow("detected circles", IMG_COLOR)
        cmd = cv.waitKey(0)
        if cmd == ord('q'):
            break
        elif cmd == ord('d'):
            continue
    cv.destroyAllWindows()
    plt.plot(group, DETECTED_CASE, label='Reference_radius', marker = 'o')
    plt.plot(group, DETECTED_ELECTRODE, label=f'{electrode.capitalize()}_radius', marker = 'o')
    plt.legend()
    plt.xlabel(f"{electrode.capitalize()} Number")
    plt.ylabel(f"{electrode.capitalize()} Radius")
    plt.title(f"Detected Radius from {electrode}")
    plt.show()
    return 0

if __name__ == "__main__":
    main()