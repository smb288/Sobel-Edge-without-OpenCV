"""
@author: sbake
"""

import cv2
import numpy as np

# Gausian blur and sobel edge kernels
Gx = np.array(([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.float32)
Gy = np.array(([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), np.float32)
gBlur = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32)/16

# Takes in image and kernel and performs 2D convolution
def conv(inImage,inKernel):
    # Pads image with edge pixels
    inImagePad = np.pad(grayLane, (1,1), 'edge')
    
    # Creates an empty array that will be populated during convolution
    outputImage = np.empty((inImage.shape[0],inImage.shape[1]), dtype='uint8')
    
    # Iterates through image and performs convolution
    for i in range(0,inImage.shape[1]):
        for j in range(0,inImage.shape[0]):
            outputImage[j,i] = (inKernel * inImagePad[j: j+3, i: i+3]).sum()
    return outputImage


# Reads in input image as grayscale
grayLane = cv2.imread('lane.png', cv2.IMREAD_GRAYSCALE)
                

# Performs gausian blur and sobel convolutions
blurIm = conv(grayLane,gBlur)
sobelX = conv(blurIm,Gx)
sobelY = conv(sobelX,Gy)
finalSobel = np.uint8(np.sqrt(sobelX**2 + sobelY**2))


# Combines images into a 2x2 format
hStack1 = np.concatenate((grayLane,blurIm,finalSobel), axis=1)

cv2.imshow('Image Outputs', hStack1)

cv2.waitKey(0)
cv2.destroyAllWindows