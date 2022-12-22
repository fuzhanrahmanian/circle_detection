import sys, os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_COLOR = None
DETECTED_CASE = []
DETECTED_ELECTRODE = []
OBJECT_LIST = ('Anode_Drop', 'Anode_Grab', 'Cathode_Drop', 'Cathode_Grab', 'Anode_Spacer_Grab', 'Cathode_Spacer_Grab', 'Cathode_Case_Grab', 'Suction_Cup', 'Reference')

CONFIG = dict(
CAM_PORT_BOTM = 1,

CAM_PORT_TOP = 2,

Scale_Drop=0.081395,

Scale_Grab=0.05714286,

Anode_Drop=dict(name='Anode_Drop', diam=15, text_pos = (10, 460), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5),
erode_iter=1, minDist=100, param1=100, param2=12, minR=88, maxR=90),

Anode_Grab=dict(name='Anode_Grab', diam=15, text_pos = (10, 460), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5),
erode_iter=1, minDist=100, param1=100, param2=25, minR=128, maxR=130),

Cathode_Drop=dict(name='Cathode_Drop', diam=14, text_pos = (10, 460), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5), 
erode_iter=1, minDist=100, param1=100, param2=15, minR=80, maxR=83),

Cathode_Grab=dict(name='Cathode_Grab', diam=14, text_pos = (10, 460), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5), 
erode_iter=1, minDist=100, param1=120, param2=15, minR=115, maxR=118),

Separator_Drop=dict(name='Separator', diam=15.5, text_pos = (10, 420), dilate_ksize=(19,19), dilate_iter=1, erode_ksize=(15,15), 
erode_iter=1, minDist=100, param1=120, param2=20, minR=130, maxR=135),

Anode_Spacer_Grab=dict(name='Anode_Spacer', diam=15.5, text_pos = (10, 420), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5), 
erode_iter=1, minDist=100, param1=120, param2=15, minR=130, maxR=135),

Cathode_Spacer_Grab=dict(name='Cathode_Spacer', diam=15.5, text_pos = (10, 420), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5), 
erode_iter=1, minDist=100, param1=120, param2=15, minR=130, maxR=135),

Cathode_Case_Grab=dict(name='Cathode_Case', diam=19.3, text_pos = (10, 420), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5), 
erode_iter=1, minDist=100, param1=120, param2=15, minR=187, maxR=189),

Reference=dict(name='Reference', diam=2, text_pos = (10, 440), dilate_ksize=(19,19), dilate_iter=1, erode_ksize=(19,19), 
erode_iter=1, minDist=100, param1=120, param2=20, minR=8, maxR=15),

Suction_Cup=dict(name='Suction_Cup', diam=4, text_pos = (10, 420), dilate_ksize=(5,5), dilate_iter=1, erode_ksize=(5,5), 
erode_iter=1, minDist=500, param1=120, param2=20, minR=54, maxR=56),

Customize=dict(name='Customize', diam=2, text_pos = (10, 420), dilate_ksize=(15,15), dilate_iter=1, erode_ksize=(15,15), 
erode_iter=1, minDist=100, param1=120, param2=5, minR=10, maxR=12),
)

def detect_object_center(object_config:dict):
    global IMG_COLOR

    img_gray = cv.cvtColor(IMG_COLOR, cv.COLOR_BGR2GRAY)
    #IMG_GRAY = cv.GaussianBlur(IMG_GRAY, (11,11), 0)
    img_gray = cv.medianBlur(img_gray, 5)
    #IMG_GRAY = cv.adaptiveThreshold(IMG_GRAY,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3.5)
    #kernel = np.ones((2,2),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, object_config['dilate_ksize']) # (19,19)
    kernelfine = cv.getStructuringElement(cv.MORPH_RECT, object_config['erode_ksize']) # (15,15)
    img_gray = cv.erode(img_gray, kernelfine, iterations=object_config['erode_iter']) # 1
    img_gray = cv.dilate(img_gray, kernel, iterations=object_config['dilate_iter']) # 1

    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, object_config['minDist'],
                               param1=object_config['param1'], param2=object_config['param2'],
                               minRadius=object_config['minR'], maxRadius=object_config['maxR'])
    (h, w) = IMG_COLOR.shape[:2]
    imageCenter = (w//2, h//2)
    # cv.line(img=IMG_COLOR, pt1=(imageCenter[0]-5, imageCenter[1]), pt2=(imageCenter[0]+5, imageCenter[1]), color=(0, 255, 0), thickness=1, lineType=8, shift=0)
    # cv.line(img=IMG_COLOR, pt1=(imageCenter[0], imageCenter[1]-5), pt2=(imageCenter[0], imageCenter[1]+5), color=(0, 255, 0), thickness=1, lineType=8, shift=0)
            
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
            (h, w) = IMG_COLOR.shape[:2]
            # scale = round(object_config['diam']/(2*radius), 6)
            # offX = round((i[0]-w//2)*scale, 4)
            # offY = round((h//2-i[1])*scale, 4) # Offset in reference to the imgcenter, inversed y axis
            offX = i[0]-w//2
            offY = i[0]-h//2
            offSet = (offX, offY) 
            cv.circle(IMG_COLOR, center, 2, (0,0,255), -1)
            cv.circle(IMG_COLOR, center, radius, (255, 0, 255), 1)
            print(f"Coordinates of {object_config['name']}: {center[0]} , {center[1]}. Radius: {i[2]}")
            cv.putText(IMG_COLOR, f"{center} R:{i[2]}", np.add(center,(10,20)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # cv.putText(IMG_COLOR, f"Offset: {offSet}px", np.add(object_config['text_pos'], (0,20)),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
def main():
    os.chdir(os.path.dirname(__file__))
    print("Choose from following list: ")
    for nr, item in enumerate(OBJECT_LIST):
        print(f"[{nr+1}]--> {item}")
    object_id = input("-------------------------\nObject to test: ")
    try:
        object_id = int(object_id)
    except ValueError:
        exit()
    else:
        object_id = OBJECT_LIST[int(object_id)-1]
    folder = os.path.join(r"trial_anode", object_id)
    group = []
    for file in os.listdir(folder):
        group.append(re.findall(r'\d+', file)[0])
        # join the path and filename
        path = os.path.join(folder, file)
        global IMG_COLOR
        IMG_COLOR = cv.imread(cv.samples.findFile(path), cv.IMREAD_COLOR)
        print(file)
        # Check if image is loaded fine
        if IMG_COLOR is None:
            print ('Error opening image!')
            sys.exit()
        detect_object_center(CONFIG[object_id])
        if object_id in ('Anode', 'Cathode'):
            detect_object_center(CONFIG['Reference'])

        cv.imshow("detected circles", IMG_COLOR)
        cmd = cv.waitKey(0)
        if cmd == ord('q'):
            break
        elif cmd == ord('d'):
            continue
    cv.destroyAllWindows()
    # plt.plot(group, DETECTED_CASE, label='Reference_radius', marker = 'o')
    # plt.plot(group, DETECTED_ELECTRODE, label=f'{object_id}_radius', marker = 'o')
    # plt.legend()
    # plt.xlabel(f"{object_id} Number")
    # plt.ylabel(f"{object_id} Radius")
    # plt.title(f"Detected Radius from {object_id}")
    # plt.show()
    # return 0

if __name__ == "__main__":
    main()