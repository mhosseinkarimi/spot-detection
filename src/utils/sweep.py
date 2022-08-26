from typing import List
import numpy as np

def sweep(srcShape, steps, endpoint=True) -> List:
    """Sweeping windows over and image to perform localized operations.

    Parameters
    ----------
    srcShape : Tuple 
        Shape of the array
    steps : List of tuple or array of integers
        steps of sweeping window in each dimension. Number of the elements should
        be lesser or equal to array dimension
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
        assert len(srcShape) > ndim
    except:
        raise ValueError("Specified steps are more than the source dimensions")
    
    sweepSteps = []
    for i in range(ndim):
        sweepSteps.append(np.arange(0, srcShape[i], steps[i]))
        if endpoint and (srcShape[i] - 1) != sweepSteps[i][-1]:
            sweepSteps[i] = np.append(sweepSteps[i], srcShape[i]-1)
    return sweepSteps

if __name__ == "__main__":
    # Test case: [array([   0,  200,  400,  600,  800, 1000, 1200, 1400, 1600, 1622]), array([   0,  300,  600,  900, 1200, 1454])]
    print(sweep((1623, 1455, 3), [200, 300]))
    # Test case: [array([   0,  200,  400,  600,  800, 1000, 1200, 1400, 1600]), array([   0,  300,  600,  900, 1200])]
    print(sweep((1623, 1455, 3), [200, 300], endpoint=False))
