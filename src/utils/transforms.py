import numpy as np
import cv2 as cv

def changeColorSpace(img, cspace="gray") -> np.ndarray:
    """Converts RGB input image in other color spaces.

    Parameters
    ----------
    img : 2D array like object
        
    cspace : str, optional
        The color space of output image. Options are: gray and HSV, by default "gray"

    Returns
    -------
    np.ndarray
        image in the new color space

    Raises
    ------
    ValueError
        If the chosen color space is not one of gray or HSV
    """
    # Converting all the forms of spelling to lower
    cspace = cspace.lower()

    # convertion
    if cspace == "gray":
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif cspace == "hsv":
        return cv.cvtColor(img, cv.COLOR_BGR2HSV)
    else:
        raise ValueError(f"{cspace} is not a valid option for color space")

def gammaCorrection(img, gamma) -> np.ndarray:
    """Gamma correction for contrast and light intensity adjustments.

    Parameters
    ----------
    img : 2D array like object
        Input image
    gamma : float
        Gamma is the non linear scaling parameter for increase or decrease each pixel's
        value in the desired scale. Gamma values lesser than 1 cause decreasing and values
        greater than 1 cause increase of pixels intensity

    Returns
    -------
    np.ndarray
        Output image, transformed using gamma correction
    """
    # Look up table
    lut = np.array([np.clip(pow(i / 255, gamma) * 255, 0, 255).astype(np.uint8) for i in range(256)])
    # Applying transformation
    img_corr = cv.LUT(img, lut)
    return img_corr


