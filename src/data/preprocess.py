import cv2 as cv
import numpy as np
from src.utils.filter import blur
from src.utils.transform import changeColorSpace, gammaCorrection, nonlinear_scaling


def initialPreprocessing(img, **kwargs) -> np.ndarray:
    """Initial preprocessings, including converting image to gray scale,
    gamma correction and sharpenning the edges.

    Parameters
    ----------
    img : A two or three dimensional array
        Source image

    Returns
    -------
    np.ndarray
        Preprocessed image
    """
    if "cspace" in kwargs.keys() and kwargs["cspace"] is not None:
        cspace = kwargs["cspace"]
    else:
        cspace = "gray"
    grayImg = changeColorSpace(img, cspace) # converting to gray scale
    gamma = kwargs["gamma"] if "gamma" in kwargs.keys() else 1.3 # setting gamma
    grayImg = gammaCorrection(grayImg, gamma) # gamma correction
    # grayImg = blur(grayImg) # applying blur
    # filtering with a shapenning filter
    kernel = np.array([[-1, -1, -1], 
                        [-1, 9, -1], 
                        [-1, -1, -1]])
    grayImg = cv.filter2D(grayImg, -1, kernel)

    return grayImg
