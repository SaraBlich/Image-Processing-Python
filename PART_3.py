# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:15:33 2019

@author: Sara
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage import io
from skimage import color

def histogram(signal, bins=100):
    max_values = np.max(signal)
    min_values = np.min(signal)
    step = (max_values-min_values)/bins
    
    values, counts = np.unique(signal, return_counts = True)
    hist = np.zeros(bins)

    for i in range(bins):
        indicis = np.logical_and(values>=step*i, values<(i+1)*step)
        temp_values = np.sum(counts[indicis])
        hist[i] = temp_values
    xbins = np.linspace(max_values, min_values, bins-1)
    return xbins, hist

def our_histogram_equalization(image, bins=100):
    hist, n_bins = np.histogram(image.ravel(), bins=bins)
    cdf = np.cumsum(hist)
    max_cdf = cdf/cdf[-1]
    new_image = np.interp(image.ravel(), n_bins[:-1], max_cdf).reshape(image.shape)
    return new_image

def run():
    path = 'C:\\Users\\Sara\\Desktop\\KOLOS_TOM\\wiki.jpg'
    image = io.imread(path)
    image_true = color.rgb2gray(image)
    image_true = image*0.5 + 0.2
    equalized_image = our_histogram_equalization(image_true, 100)
    
    fft_image = np.fft.fft2(image)

    hist, bins = histogram(image,30)
    plt.figure()
    plt.plot(hist,bins[:-1],'r*-')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax =1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run()
                                  
