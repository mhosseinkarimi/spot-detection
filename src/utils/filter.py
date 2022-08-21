import numpy as np
import cv2 as cv

def blur(img, method="gaussian", kernel_size=5, sigma=1, sigmaColor=1) -> np.ndarray:
    """Blurs the image and reduces the effect of noise and outliers.

    Parameters
    ----------
    img : 2D array like object
        Input image
    method : str, optional
        Method of blurring; it can be selected between gaussian, average, median, bilateral, by default "gaussian"
    kernel_size : int, optional
        Size of the blurring kernel, by default 5
    sigma : int, optional
        The standard deviation of gaussian kernel. 
        Only used when using gaussian or bilateral method and is ignored otherwise, by default 1
    sigmaColor : int, optional
        Standard deviation of gaussian kernel in color space.
        Only used in bilateral method, by default 1

    Returns
    -------
    np.ndarray
        Blurred image

    Raises
    ------
    ValueError
        If the specified method is not one of the valid options
    """
    # Converting all the forms of spelling to lower
    method = method.lower()

    # applying selected blurring methods
    if method == "gaussian":
        return cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    elif method == "average":
        return cv.blur(img, (kernel_size, kernel_size), borderType=cv.BORDER_REFLECT)
    elif method == "median":
        return cv.medianBlur(img, kernel_size)
    elif method == "bilateral":
        return cv.bilateralBlur(img, kernel_size, sigmaColor, sigma)
    else:
        raise ValueError(f"{method} is not a valid option for blurring method")

    