from typing import List
import numpy as np

def sweepStep(srcShape, steps, offset=0, endpoint=True) -> List:
    """Sweeping windows over and image to perform localized operations.

    Parameters
    ----------
    srcShape : Tuple 
        Shape of the array
    steps : List of tuple or array of integers
        steps of sweeping window in each dimension. Number of the elements should
        be lesser or equal to array dimension
    offset : int 
        The offset applied to the start of signal for windows. Negative offsets mean shifting the window 
        to the left and possitive offsets mean shifting window to right.
    endpoint : bool, optional
        Choice of including or excluding end points at each axis, by default True

    Returns
    -------
    List
        The list of steps at each dimension of array

    Raises
    ------
    ValueError
        If the number of step sizes are grater than array dimensions, ValueError is raised
    """
    ndim = len(steps)
    try:
        assert len(srcShape) >= ndim
    except:
        raise ValueError("Specified steps are more than the source dimensions")
    
    sweepSteps = []
    for i in range(ndim):
        tempSteps = np.arange(offset, srcShape[i], steps[i], dtype=int)
        if offset < 0:
            tempSteps = tempSteps[tempSteps > 0] 
            tempSteps = np.insert(tempSteps, 0, 0).astype(int)
        sweepSteps.append(tempSteps)
        if endpoint and (srcShape[i] - 1) != sweepSteps[i][-1]:
            sweepSteps[i] = np.append(sweepSteps[i], srcShape[i]-1)
    return sweepSteps

if __name__ == "__main__":
    # Test case: [array([  30,  230,  430,  630,  830, 1030, 1230, 1430, 1622]), array([  30,  330,  630,  930, 1230, 1454])]
    print(sweepStep((1623, 1455, 3), [200, 300], offset=30, endpoint=True))
    # Test case: [array([   0,  196,  396,  596,  796,  996, 1196, 1396, 1596]), array([   0,  296,  596,  896, 1196])]
    print(sweepStep((1623, 1455, 3), [200, 300], offset=-4, endpoint=False))
