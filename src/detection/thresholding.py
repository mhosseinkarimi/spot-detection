import cv2 as cv
import numpy as np
from src.utils.windowing import *

def globalThresholding(
    img, 
    threshmethod=cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
    initThresh=127,
    ) -> np.ndarray:
    """Global thresholding with appointed methods

    Parameters
    ----------
    img : A two or three dimensional array
        Source image
    threshmethod : int, optional
        Defines the method of the thresholding, by default cv.THRESH_BINARY_INV+cv.THRESH_OTSU
    initThresh : int, optional
        Initial value for threshold function
    Returns
    -------
    np.ndarray
        Thresholded image
    """
    ret, threshImg = cv.threshold(img, initThresh, 255, threshmethod) # thresholding
    return threshImg


def globalAdaptiveThresholding(
    img, 
    threshmethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshType=cv.THRESH_BINARY_INV,
    blockSize=11
) -> np.ndarray:
    """Adaptive global thresholding.

    Parameters
    ----------
    img : A two or three dimensional array
        Source image
    threshmethod : int, optional
        Defines the algorthm for thresholding, by default cv.ADAPTIVE_THRESH_GAUSSIAN_C
    threshType : int, optional
        Defines the type of binarization, by default cv.THRESH_BINARY_INV

    Returns
    -------
    np.ndarray
        Thresholded image
    """
    threshImg = cv.adaptiveThreshold(
        img, 255, threshmethod, threshType, blockSize, 2
        )
    return threshImg


def localThresholding(
    img, 
    threshmethod=cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
    steps=(100, 100),
    offset=0,
    initThresh=127
    ) -> np.ndarray:
    ySteps, xSteps = sweepingWindow(img.shape, steps, offset) # creating the windows

    threshImg = np.zeros_like(img)
    for i in range(len(xSteps)-1):
        for j in range(len(ySteps)-1):
            cutImg = img[ySteps[j]:ySteps[j+1], xSteps[i]:xSteps[i+1]]
            ret, threshImg[ySteps[j]:ySteps[j+1], xSteps[i]:xSteps[i+1]] = \
            cv.threshold(cutImg, initThresh, 255, threshmethod) # thresholding
    return threshImg


def localAdaptiveThresholding(
    img, 
    threshmethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshType=cv.THRESH_BINARY_INV,
    blockSize=11,
    steps=(100, 100),
    offset=0,
) -> np.ndarray:
    ySteps, xSteps = sweepingWindow(img.shape, steps, offset) # creating the windows

    threshImg = np.zeros_like(img)
    for i in range(len(xSteps)-1):
        for j in range(len(ySteps)-1):
            cutImg = img[ySteps[j]:ySteps[j+1], xSteps[i]:xSteps[i+1]]
            threshImg[ySteps[j]:ySteps[j+1], xSteps[i]:xSteps[i+1]] = cv.adaptiveThreshold(
                cutImg, 255, threshmethod, threshType, blockSize, 2
                ) # thresholding
    return threshImg

