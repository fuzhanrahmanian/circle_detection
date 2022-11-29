import cv2
import numpy as np
#pylint: disable=no-member
img = cv2.imread('trial_anode/[No55]_Anode_2022_11_18_13h_42m_38s.jpg', 0)
#img = cv2.threshold(img, 62, 255, cv2.THRESH_BINARY)[1]  # ensure binary
# invert the binary image
#img = cv2.bitwise_not(img)
#cv2.imwrite('inverted_binary_white.png', img)
#num_labels, labels_im = cv2.connectedComponents(img)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imwrite('labeled.png', labeled_img)
    cv2.waitKey()


def edge_detection(image):
    
    fgbg = cv2.createBackgroundSubtractorMOG2(
    history=10,
    varThreshold=2,
    detectShadows=False)
    # Converting the image to grayscale.
    gray = image#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract the foreground
    edges_foreground = cv2.bilateralFilter(gray, 9, 175, 75)
    foreground = fgbg.apply(edges_foreground)
    
    # Smooth out to get the moving area
    kernel = np.ones((50,50),np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Applying static edge extraction
    edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    edges_filtered = cv2.Canny(edges_foreground, 10, 35)

    # Crop off the edges out of the moving area
    cropped = (foreground // 255) * edges_filtered

    # Stacking the images to print them together for comparison
    images = np.hstack((gray, edges_filtered, cropped))
    cv2.imwrite('edge_detection.png', images)



edge_detection(img)
#imshow_components(labels_im)