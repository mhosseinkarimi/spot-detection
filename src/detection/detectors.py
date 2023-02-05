import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from abc import ABC, abstractclassmethod
from src.detection.thresholding import *
from src.data.preprocess import initialPreprocessing
from src.detection.contouring import contourWithStats
from src.detection.watershed import watershed


class BaseDetector(ABC):
    @abstractclassmethod
    def detect(self, img, **kwargs):
        pass
    

    @abstractclassmethod
    def stats(self, *args, **kwargs):
        pass


class ThresholdDetector(BaseDetector):
    def __init__(self, thresholdMethod="global", threshSteps=None, **threshMethodkwargs) -> None:
        self.steps = threshSteps
        if thresholdMethod == "global":
            self.method = globalThresholding
        elif thresholdMethod == "global_adaptive":
            self.method = globalAdaptiveThresholding
        elif thresholdMethod == "local":
            self.method = localThresholding
        elif thresholdMethod == "local_adaptive":
            self.method = localAdaptiveThresholding
        else:
            raise ValueError("Invalid thresholing method.")
        self.methodkwargs = threshMethodkwargs
    

    def detect(self, img):
        grayImg = initialPreprocessing(img)
        if self.steps is not None:
            threshImg = self.method(grayImg, steps=self.steps)
        else:
            threshImg = self.method(grayImg)
        return threshImg
    

    def stats(self, threshImg, h_cm, w_cm, hist_plot=False, print_results=False):
        cents, area, perimeter = contourWithStats(threshImg, showContours=False)
        numSpots = len(cents) # number of spots
        density = numSpots / (h_cm * w_cm)
        h, w = threshImg.shape[:2] # hight and width of image in pixels
        pixelArea = h_cm * w_cm / (h * w)
        diam = np.sqrt(np.array(area) * pixelArea / np.pi) * 2e4 # diameter in um
        meanDiam = np.mean(diam)
        medianDiam = np.median(diam)
        minDiam = np.min(diam)
        maxDiam = np.max(diam)

        if hist_plot:
            plt.hist(diam, bins=np.arange(0, np.max(diam), 100))
            plt.xlabel("Diameter (um)")
            plt.ylabel("Count")
            plt.title("Distribution of diameters")
            plt.show()
        
        if print_results:
            print("==================================================")
            print(f"Number of spots: {numSpots}")
            print(f"Density: {density:.2} spots per squared cm")
            print(f"Mean diameter: {meanDiam:.2f} um")
            print(f"Median diam: {medianDiam:.2f} um")
            print(f"Minimum diameter: {minDiam:.2f} um")
            print(f"Maximum diameter: {maxDiam:.2f} um")

        stats_param = {
            "num_spots": numSpots,
            "density": density,
            "mean_diam": meanDiam,
            "median_diam": medianDiam,
            "min_diam": minDiam,
            "max_diam":maxDiam,
        }
        return stats_param


class WatershedDetector(BaseDetector):
    def __init__(self, thresholdMethod="local", thresholdStep=None) -> None:
        if thresholdMethod == "global":
            self.thresholdMethod = globalThresholding
        elif thresholdMethod == "global_adaptive":
            self.thresholdMethod = globalAdaptiveThresholding
        elif thresholdMethod == "local":
            self.thresholdMethod = localThresholding
        elif thresholdMethod == "local_adaptive":
            self.thresholdMethod = localAdaptiveThresholding
        else:
            raise ValueError("Invalid thresholing method.")

        if thresholdStep and "local" in thresholdMethod:
            self.thresholdStep = thresholdStep
        else:
            self.thresholdStep = (300, 300)
    
    
    def detect(self, img):
        grayImg = initialPreprocessing(img)
        threshImg = self.thresholdMethod(grayImg) 
        labels = watershed(threshImg)
        return labels
    

    def stats(self, labels, h_cm, w_cm, hist_plot=False, print_results=False):
        numSpots = np.max(np.unique(labels))
        density = numSpots / (h_cm * w_cm)
        h, w = labels.shape[:2]
        pixelArea = h_cm * w_cm / (h * w)
        spotsLabels = np.unique(labels)[1:]
        area = []
        for label in spotsLabels:
            area.append(np.count_nonzero(labels == label))
        diameter = np.sqrt(np.array(area) * pixelArea / np.pi) * 2e4
        meanDiam = np.mean(diameter)
        medianDiam = np.median(diameter)
        minDiam = np.min(diameter)
        maxDiam = np.max(diameter)

        if hist_plot:
            plt.hist(diameter, bins=np.arange(0, np.max(diameter), 100))
            plt.xlabel("Diameter (um)")
            plt.ylabel("Count")
            plt.title("Distribution of diameters")
            plt.show()

        if print_results:
            print("==================================================")
            print(f"Number of spots: {numSpots}")
            print(f"Density: {density} spots per squared cm")
            print(f"Mean diameter: {meanDiam:.2f} um")
            print(f"Median diam: {medianDiam:.2f} um")
            print(f"Minimum diameter: {minDiam:.2f} um")
            print(f"Maximum diameter: {maxDiam:.2f} um")

        stats_param = {
            "num_spots": numSpots,
            "density": density,
            "mean_diam": meanDiam,
            "median_diam": medianDiam,
            "min_diam": minDiam,
            "max_diam":maxDiam,
        }
        return stats_param
