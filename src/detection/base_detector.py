import numpy as np
import cv2 as cv
from abc import ABC, abstractclassmethod

class BaseDetector(ABC):
    """Base Detector class for spot detection.
    """
    @abstractclassmethod
    def detect(self, img, show=False, **kwargs):
        pass
    
    @abstractclassmethod
    def stats(self, img_size_cm, stats):
        pass
