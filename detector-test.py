import cv2 as cv
import numpy as np
from src.detection.detectors import WatershedDetector, ThresholdDetector

img = cv.imread("/mnt/c/Users/mhuss/OneDrive/Desktop/test_images/400um/1.jpg")
img_cut = img[100:500, :150]
# detector = ThresholdDetector("local", (50, 50))
detector = WatershedDetector("local", (50, 50))

label = detector.detect(img_cut)
detector.stats(label, 3.1, 1.26, print_results=True)

back = np.ones(label.shape+(3,)) * 255
for i in range(1, np.max(label)):
    color = np.random.randint(0, 256, size=(3,), dtype="uint8")
    back[label == i] = color

cv.imwrite("/mnt/c/Users/mhuss/OneDrive/Desktop/img-cut.jpg", img_cut)
cv.imwrite("/mnt/c/Users/mhuss/OneDrive/Desktop/water-test.jpg", back)