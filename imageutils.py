#import cv2
from skimage.transform import resize
import numpy as np


def normalize_image(img, target_size):
    # resize
    scale_x = target_size[0]/img.shape[0]
    scale_y = target_size[1]/img.shape[1]
    scale = min(scale_x, scale_y, 1)
    
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    resized = resize(img, dim)
    
    # zeropad
    x_before = round((target_size[0]-resized.shape[0])/2+0.1)
    x_after = round((target_size[0]-resized.shape[0])/2-0.1)
    y_before = round((target_size[1]-resized.shape[1])/2+0.1)
    y_after = round((target_size[1]-resized.shape[1])/2-0.1)
    
    padded = np.pad(resized,((x_before,x_after),(y_before,y_after),(0,0)),'constant', constant_values=0)
    
    return(padded)

