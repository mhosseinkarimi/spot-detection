import cv2 as cv 
import numpy as np
from src.detection.watershed import WatershedDetector
#
img_path = "/home/curiouscoder/Downloads/spot detection project material/test1_720.jpeg"
# img_path = "/home/curiouscoder/Desktop/spraying project/test 1/fig1.jpg"
img = cv.imread(img_path)

detector = WatershedDetector(sweep=True, threshStep=(400, 400), watershedStep=(50, 50))
cents, stats = detector.detect(img, verbose=1, save_path="/home/curiouscoder/Desktop/test1_res.jpg")
detector.stats(
    img_size_cm=(6.6, 2.1), img_size_pxl=img.shape[:2],
    stats=stats, show_results=True
)