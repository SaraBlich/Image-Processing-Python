# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:48:31 2019

@author: Sara
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

def generate_circle(x_size, y_size, x_origin, y_origin, radius):
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    
    image = np.square(grid_x - x_origin) + np.square(grid_y - y_origin) < radius**2
    return image

def generate_rectangle(x_size, y_size, x_b, y_b, width, height):
    image = np.zeros((y_size, x_size))
    image[y_b:y_b + height, x_b:x_b + width] = 1
    
    return image

def dilatation(image, se):
    return (nd.generic_filter(image, lambda a: np.max(a), footprint = se.T))

def erosion(image, se):
    return (nd.generic_filter(image, lambda a: np.min(a), footprint = se.T))


def run():
    
    
    image_true = generate_circle(200, 200, 50, 50, 30)
    plt.figure()
    plt.imshow(image_true, cmap='gray')
    
    image_new = dilatation(image_true, np.ones((10, 10))) 
    plt.figure()
    plt.imshow(image_new, cmap='gray')
    plt.show()
    
    image_1 = generate_circle(128, 128, 64, 64, 30)
    image_2 = generate_circle(128, 128, 64, 64, 15)
    
    image = np.logical_and(image_1, np.logical_not(image_2))
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()
    
if __name__ == "__main__":
    run()