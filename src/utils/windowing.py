from typing import List

import numpy as np


def sweepingWindow(imgShape, steps, offset, endpoint=True) -> List[np.ndarray]:
    ndim = len(steps) 
    # asserting that the appointed steps matches image dimensions
    try:
        assert len(imgShape) >= ndim 
    except:
        raise ValueError("Specified steps are more than the source dimensions")
    
    # generating steps of sweeping window
    sweepSteps = []
    for i in range(ndim):
        sweepSteps.append(np.arange(offset, imgShape[i], steps[i]))
        if offset < 0:
            sweepSteps[i] = sweepSteps[i][sweepSteps[i] > 0]
            sweepSteps[i] = np.insert(sweepSteps[i], 0, 0) # Adding 0 to the beginning
        if endpoint and (imgShape[i] - 1) != sweepSteps[i][-1]:
            sweepSteps[i] = np.append(sweepSteps[i], imgShape[i]-1)
    return sweepSteps


def multipleSweepingWindow(
    imgShape, steps, offsets, endpoint=True
    ) -> List[List[np.ndarray]]:
    sweepSteps = []
    # generating sweeping windows for differnt offsets
    for offset in offsets:
        sweepSteps.append(sweepingWindow(imgShape, steps, offset, endpoint))
    return sweepSteps
