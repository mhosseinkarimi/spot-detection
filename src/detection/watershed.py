import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as ski_watershed


def watershed(threshImg) -> np.ndarray:
   distance = ndimage.distance_transform_edt(threshImg) # distance transform
   localMax = peak_local_max(distance, min_distance=6, footprint=np.ones((3, 3)),
   labels=threshImg) # finding local maximas
   boolMask = np.zeros_like(distance, dtype=bool) # boolean mask
   boolMask[tuple(localMax.T)] = True
   markers, _ = ndimage.label(boolMask, structure=np.ones((3, 3))) # label structures 
   labels = ski_watershed(-distance, markers, mask=threshImg) # watershed
   return labels

