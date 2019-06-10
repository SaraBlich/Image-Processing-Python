# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:27:00 2019

@author: Sara
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd



#IMAGE EROSION - made step by step
def my_erosion(image, se):
    output_image = np.zeros(image.shape)
    size_y, size_x = image.shape
    se_size_y, se_size_x = se.shape
    
    padding_y = int((se_size_y-1)/2)
    padding_x = int((se_size_x-1)/2)
    
    for j in range(size_y):
        for i in range(size_x):
            b_y = max(j - padding_y, 0)
            e_y = min(j + padding_y + 1, size_y - 1)
            b_x = max(i - padding_x, 0)
            e_x = min(i + padding_x + 1, size_x - 1)
            temp_patch = image[b_y:e_y, b_x:e_x]
            temp_value = np.min(temp_patch)
            output_image[j, i] = temp_value
    return output_image

#IMAGE DILATION - made step by step
def my_dilation(image, se):
    output_image = np.zeros(image.shape)
    size_y, size_x = image.shape
    se_size_y, se_size_x = se.shape
        
    padding_y = int((se_size_y-1)/2)
    padding_x = int((se_size_x-1)/2)
        
    for j in range(size_y):
        for i in range(size_x):
            b_y = max(j - padding_y, 0)
            e_y = min(j + padding_y + 1, size_y - 1)
            b_x = max(i - padding_x, 0)
            e_x = min(i + padding_x + 1, size_x - 1)
            temp_patch = image[b_y:e_y, b_x:e_x]
            temp_value = np.max(temp_patch)
            output_image[j, i] = temp_value
    return output_image

#IMAGE EROSION - using scipy.ndimage library methods
def erosion(image, se):
    return nd.generic_filter(image, lambda a: np.min(a), footprint=se.T)

#IMAGE DILATION - using scipy.ndimage library methods
def dilation(image, se):
    return nd.generic_filter(image, lambda a:np.max(a), footprint = se.T)

#IMAGE OPENING AS A COMBINATION OF SCIPY.NDIMAGE LIBRARY METHODS
def opening(image, se):
    return dilation(erosion(image, se), se)

#IMAGE CLOSING AS A COMBINATION OF SCIPY.NDIMAGE LIBRARY METHODS
def closing(image, se):
    return erosion(dilation(image, se), se)
    

def run():
    
# Checking out how my_dilation and my_erosion works (we wont be trying ready functions written above)
    
    image = np.zeros((128, 128))
    image[30:80, 40:65] = 1
    se = np.ones((15,15))
    
    eroded_image= my_erosion(image, se)
    dilated_image = my_dilation(image, se)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(eroded_image, cmap='gray')
    plt.axis('off')
    plt.title('Erosion')
    plt.subplot(1, 2, 2)
    plt.imshow(dilated_image, cmap='gray')
    plt.axis('off')
    plt.title('Dilation')
    plt.show()
    
# Checking out how opening and closing works (also we will add some noise)
    image = np.zeros((128, 128))
    image[30:80, 40:65] = 1
    image[45:50, 47:51] = 0
    
    se = np.ones((3, 3))

# Adding some noise by using random numbers from normal distribution
    noise = np.random.randn(128, 128)
    
    binary_noise = np.abs(noise) >2
    image_with_noise = np.logical_or(image, binary_noise)    
    opened_image = opening(image_with_noise, np.ones((3, 3)))  
    contour_1 = np.logical_and(image, np.logical_not(erosion(image, se)))	
    contour_2 = np.logical_and(dilation(image, se), np.logical_not(image))
    closed_image = closing(image, np.ones((21, 21)))

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Generated image')
    plt.subplot(2, 3, 2)
    plt.imshow(binary_noise, cmap='gray')
    plt.axis('off')
    plt.title('Binary noise')
    plt.subplot(2, 3, 3)
    plt.imshow(image_with_noise, cmap='gray')
    plt.title('Image with noise')
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(opened_image, cmap='gray')
    plt.axis('off')
    plt.title('Opened image')
    plt.subplot(2, 2, 2)
    plt.imshow(contour_1, cmap = 'gray')
    plt.axis('off')
    plt.title('Contour 1')
    plt.show()
    
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(contour_2, cmap='gray')
    plt.title('Contour 2')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(closed_image, cmap='gray')
    plt.axis('off')
    plt.title('Closed image')
    plt.show()    
    
    
if __name__ == "__main__":
    run()