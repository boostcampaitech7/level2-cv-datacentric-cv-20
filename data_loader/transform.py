import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

def get_train_transform():
    return A.Compose([
        # Sharpen(),
        A.ColorJitter(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])

def get_val_transform():
    return A.Compose([
        A.ColorJitter(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
    ])

class Sharpen(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Sharpen, self).__init__(always_apply=always_apply, p=p)
        self.kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    
    def apply(self, image, **params):
        sharpened = cv2.filter2D(image, -1, self.kernel)
        return sharpened
