import cv2 as cv
import numpy as np

def contourWithStats(
    img, retrMode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE, showContours=False):
    # finding contours 
    contours, _ = cv.findContours(img, mode=retrMode, method=method) 
    cents = []
    area = []
    perimeter = []
    # calculating properties
    for cont in contours:
        M = cv.moments(cont)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cents.append((cx, cy))
            area.append(cv.contourArea(cont))
            perimeter.append(cv.arcLength(cont, True))
    # viewing the detected contours
    if showContours:
        cont_mask = 255 * np.ones_like(img).astype("uint8")
        cv.drawContours(cont_mask, contours, -1, color=0, thickness=1)
        cv.imshow("Contours", cont_mask)
        cv.waitKey(0)
    return cents, area, perimeter