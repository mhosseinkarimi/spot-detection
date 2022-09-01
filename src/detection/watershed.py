from typing import Tuple

import cv2 as cv
import numpy as np
from src.utils.filter import blur
from src.utils.sweep import sweepStep
from src.utils.transforms import changeColorSpace, gammaCorrection

from .base_detector import BaseDetector


class WatershedDetector(BaseDetector):
    def __init__(
        self, thresholdType=None, sweep=False, 
        threshStep=(500, 500), watershedStep=(300, 300)) -> None:
        """WatershedDetector constructor.

        Parameters
        ----------
        thresholdType : int or None, optional
            Type of thresholding based on cv2 standards if None,
            Otsu and Inverse binary threshold would be applied, by default None
        sweep : bool, optional
            Defines wether to apply the detection on smaller windows or globally, by default False
        threshStep : tuple, optional
            Steps of sweeping window in each dimension with thresholding, by default (500, 500)
        watershedStep : tuple, optional
            Steps of sweeping window in each dimension in watershed algorithm, by default (300, 300)

        Raises
        ------
        ValueError
            If the thresholdType is not an integer ValueError will be raised
        """
        self.sweep = sweep
        if thresholdType is None:
            self.thresholdType = cv.THRESH_BINARY_INV+cv.THRESH_OTSU
        elif not(isinstance(thresholdType, int)):
            raise ValueError("Threshold types should be from int type")
        else:
            self.thresholdType = thresholdType
        self.threshStep = threshStep
        self.watershedStep = watershedStep
    
    def _get_steps(self, imgShape) -> Tuple[np.ndarray]:
        """Calculating the sweeping steps.

        Parameters
        ----------
        imgShape : Tuple of integers
            _Shape of input image

        Returns
        -------
        Tuple[np.ndarray]
            A tuple containing the steps for each dimension
        """
        threshSteps = sweepStep(imgShape, self.threshStep, endpoint=True)
        watershedSteps = sweepStep(imgShape, self.watershedStep, endpoint=True)
        return threshSteps, watershedSteps
    
    def threshold(self, img, threshSteps=None) -> np.ndarray:
        """Thresholding algorithm with the option of sweeping on the image to provide
        localization.

        Parameters
        ----------
        img : Gray scale image a 2-D array
            Input image
        threshSteps : Tuple of int, optional
            Length and width of the sweeping window, by default None. If it's not passes
            the algorithm is applied globally

        Returns
        -------
        np.ndarray
            Thresholded binary image

        Raises
        ------
        ValueError
            Error is raised if image has more than one channel 
        """
        # Asserting that image is in gray scale
        try:
            assert img.ndim < 3
        except:
            raise ValueError("Image must be in gray scale or binary")
        
        # The case of sweeping 
        if self.sweep and threshSteps is not None:
            threshImg = np.zeros(img.shape)
            # Thresholding
            xSteps, ySteps = threshSteps
            for i in range(len(xSteps)-1):
                for j in range(len(ySteps)-1):
                    cutImg = img[xSteps[i]:xSteps[i+1], ySteps[j]:ySteps[j+1]]
                    ret, thresh = cv.threshold(cutImg, 0, 255, self.thresholdType)
                    threshImg[xSteps[i]:xSteps[i+1], ySteps[j]:ySteps[j+1]] = thresh
            return threshImg
        # The case of global thresholding
        else:
            ret, threshImg = cv.threshold(img, 0, 255, self.thresholdType)
            return threshImg
    
    def watershed(self, img) -> Tuple[np.ndarray]:
        """Watershed algorithm for spot detection using thresholded images.

        Parameters
        ----------
        img : A binary image, 2-D array
            Input thresholded binary image

        Returns
        -------
        Tuple of np.ndarrays
            Returns number of detected spots, labels assigned to each spot, statistics,
            centers of the contours
        """
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN,kernel, iterations = 2).astype("uint8")
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        color_img = cv.cvtColor(img.astype("uint8"), cv.COLOR_GRAY2BGR)
        # applying watershed
        markers = cv.watershed(color_img, markers)
        markers[0, :], markers[-1, :]= 0, 0
        markers[:, 0], markers[:, -1] =0, 0
        # mask of detected contours
        mask = np.zeros(img.shape, dtype="uint8")
        mask[markers == -1] = 255

        # counting the number of contours
        numlabels, labels, stats, cents = cv.connectedComponentsWithStats(mask)
        numlabels -= 1
        stats = stats[1:]
        cents = cents[1:]
        # location of the center of the contours
        cents = np.rint(cents).astype("int")
        return numlabels, labels, stats, cents
    
    def stats(self, img_size_cm, img_size_pxl, stats, show_results=False) -> Tuple[float]:
        """Statistics of the detected spots

        Parameters
        ----------
        img_size_cm : Tuple of two floats
            Height and width of the image in cm
        img_size_pxl : Tuple of two integers
            Height and width of the image in pixels
        stats : An array contatining the stats provided by detect method
            Statistical features of detected spots
        show_results : bool, optional
           Option of printing the results at the end of the method, by default False

        Returns
        -------
        Tuple of floats
            Mean and median of areas of spots and mean and median of diameters of spots in that order

        Raises
        ------
        ValueError
            Error is raised if the image size in pixels or cm is not a tuple of 2 numbers with the correct type
        """
        # Checking for valid inputs 
        if len(img_size_pxl) > 2 or not(np.all([isinstance(x, int) for x in img_size_pxl])):
            raise ValueError("Expected a two element tuple of integers.")
        # Area of the image in pixels
        img_area_pxl = img_size_pxl[0] * img_size_pxl[1]

        # Checking for valid inputs 
        if len(img_size_cm) > 2 or not(np.all([isinstance(x, float) for x in img_size_cm])):
            raise ValueError("Expected a two element tuple of floats.")
        # Area of the image in cm^2
        img_area_cm = img_size_cm[0] * img_size_cm[1]

        # Extracting statistics of area and diameter of the spots
        num_spots = len(stats)
        spot_density = num_spots / img_area_cm
        spot_area = (stats[:, -1] / img_area_pxl) * img_area_cm
        spot_area_mean = np.mean(spot_area)
        spot_area_median = np.median(spot_area)
        spot_area_min = np.min(spot_area)
        spot_area_max = np.max(spot_area)
        spot_diameter = 2 * np.sqrt(spot_area / np.pi)
        spot_diam_mean = np.mean(spot_diameter)
        spot_diam_median = np.median(spot_diameter)
        spot_diam_min = np.min(spot_diameter)
        spot_diam_max = np.max(spot_diameter)

        if show_results:
            print(f"Density of spots in square cm: {spot_density}")
            print(f"Mean area of detected spots: {spot_area_mean:2f} squared cms")
            print(f"Median area of detected spots: {spot_area_median:2f} squread cms")
            print(f"Minimum area of detected spots: {spot_area_min:2f} squared cms")
            print(f"Maximum area of detected spots: {spot_area_max:2f} squared cms")
            print(f"Mean diameter of the detected spots: {1e4 * spot_diam_mean:2f} um")
            print(f"Median diameter of detected spots: {1e4 * spot_diam_median:2f} um")
            print(f"Minimum diameter of detected spots: {1e4 * spot_diam_min:2f} um")
            print(f"Maximum diameter of detected spots: {1e4 * spot_diam_max:2f} um")
        
        stats = {
            "density": spot_density,
            "min_area": spot_area_min,
            "max_area": spot_area_max,
            "mean_area": spot_area_mean,
            "median_area": spot_area_median,
            "min_diam": spot_diam_min,
            "max_diam": spot_diam_max,
            "mean_diam": spot_diam_mean,
            "median_diam": spot_diam_median,
        }
        return stats


    def detect(self, img, verbose=0, save_path=None,**kwargs) -> None:
        """API for using this detector.

        Parameters
        ----------
        img : Image, 2-D or 3-D array
            input image; can be colored or gray scale image
        verbose: int, Optional
            The option to wheter show the results. If set to 0 nothing will be shown, if 1 is selected
            Only the final result is shown, and if is set to 2 the final result and the result at each 
            sweeping stage would be illustrated, by default is 0
        save_path: Path like object or string
            the path to saving destination
        """

        maskedImg = img.copy()

        # converting image to gray scale
        imgGray = changeColorSpace(img, cspace="gray")
        imgGray = gammaCorrection(imgGray, gamma=1.3)
        imgGray = blur(imgGray, **kwargs)
        kernel = np.array([[-1, -1, -1], 
                           [-1, 9, -1], 
                           [-1, -1, -1]])
        imgGray = cv.filter2D(imgGray, -1, kernel)

        # Perform sweeping 
        num_points = 0
        watershed_stats = []
        
        if self.sweep:
            threshSteps, watershedSteps = self._get_steps(imgGray.shape)
            threshImg = self.threshold(imgGray, threshSteps)
            xSteps, ySteps = watershedSteps
            
            for i in range(len(xSteps)-1):
                for j in range(len(ySteps)-1):
                    # cutting a window of images
                    cutImgGray = threshImg[xSteps[i]:xSteps[i+1], ySteps[j]:ySteps[j+1]]
                    cutImg = maskedImg[xSteps[i]:xSteps[i+1], ySteps[j]:ySteps[j+1]]
                    # applying threshold
                    num_spot, spot_label, stats, cents = self.watershed(cutImgGray)
                    num_points += len(cents)
                    watershed_stats.extend(stats)
                    # marking the contours on colored image
                    cutImg[spot_label != 0] = (255, 0, 0)
                    # marking the center of each contour
                    for cnt in cents:
                        cutImg = cv.circle(cutImg, (cnt[0], cnt[1]),
                        radius=5, color=(0, 0, 255), thickness=-1)
                    if verbose == 2:
                        cv.imshow("Detected spots in segment",cutImg)
                        cv.waitKey(0)
        # Applying the algorithm globally
        else:
            threshImg = self.threshold(imgGray)
            
            num_spot, spot_label, stats, cents = self.watershed(threshImg)
            num_points += len(cents)
            # marking the contours on colored image
            threshImg[spot_label != 0] = (255, 0, 0)
            # marking the center of each contour
            for cnt in cents:
                cutImg = cv.circle(threshImg, (cnt[0], cnt[1]),
                radius=3, color=(0, 0, 255), thickness=-1)

        # Showing the final result
        if verbose > 0:
            cv.imshow("Maksed Image",maskedImg)
            cv.waitKey(0)
        # Saving the marked image
        if save_path is not None:
            cv.imwrite(save_path, maskedImg)

        print("=================Results==============")
        print(f"Number of detected spots: {num_points}")
        print(f"Marked image is saved on : {save_path}")

        return np.array(watershed_stats)
