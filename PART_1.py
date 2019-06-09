# -*- coding: utf-8 -*-
"""

Created on Sun Jun  9 20:10:48 2019

@author: Sara
"""
""" gradient, kat gradientu, kierunek gradientu, normalizacje, maski previtt itd"""

import numpy as np
import scipy.signal as signal
import scipy.ndimage as nd
import matplotlib.pyplot as plt 
from skimage import io
from skimage import color

#GRADIENT

def my_gradient(image, mode):
    gradient_x = np.zeros(image.shape)
    gradient_y = np.zeros(image.shape)
    if mode == "forward":
        gradient_x[:, 0:-1] = image[:, 1:] - image[:, 0:-1]
        gradient_y[0:-1, :] = image[1:, :] - image[0:-1, :]
    elif mode == "backward":
        gradient_x[:, 1:] = image[:, 1:] - image[:, 0:-1]
        gradient_y[1:, :] = image[1:, :] - image[0:-1, :]
    elif mode == "central":
        gradient_x[:, 1:-1] = (image[:, 2:] - image[:, 0:-2])/2
        gradient_y[1:-1, :] = (image[2:, :] - image[0:-2, :])/2
    else:
        raise ValueError("Mode not supported.")
        
    return [gradient_y, gradient_x]

#NORMALIZACJA -> OD OBRAZU ODEJMUJE MIN I DZIELE PRZEZ MAX ODJAC MIN
def normalize(image):
    return ((image - np.min(image)) / (np.max(image) - np.min(image)))

#NORMALIZACJA_Z -> OD OBRAZU ODEJMUJE SREDNIA I DZIELE PRZEZ ODCHYLENIE STANDARDOWE OBRAZU
def normalize_z(image):
    return ((image - np.mean(image))/(np.std(image)))

#FUNKCJA GENERUJACA MASKE GAUSSOWSKA I FILTRUJACA WYGENEROWANA MASKA
def gaussian(image, size, sigma):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    image_conv = signal.convolve2d(image, g)
    return image_conv

def run():
    #WCZYTYWANIE OBRAZU
    image_path = 'C:\\Users\\Sara\\Desktop\\KOLOS_TOM\\coins.png'
    image_read = io.imread(image_path)
    image_true = color.rgb2gray(image_read)
    gradient = my_gradient(image_true, "forward")
    
    #UZYSKIWANIE GRADIENT_X, GRADIENT_Y Z FUNKCJI
    gradient_y, gradient_x = gradient[0], gradient[1]
    
    #MAGNITUDA GRADIENTU
    gradient_magnitude = np.sqrt(np.square(gradient_y) + np.square(gradient_x))
    
    #KAT GRADIENTU
    gradient_angle = np.arctan2(gradient_y,gradient_x)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_read, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(gradient_x, cmap='gray')
    plt.axis('off')
    plt.show()
    
    #MASKA PREWITT
    prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
            ])
    prewitt_y = prewitt_x.T
    
    #MASKA SOBEL
    sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
            ])
    sobel_y = sobel_x.T
    
    #NAKLADANIE MASKI
    prewitt_x_im = signal.convolve2d(image_true, prewitt_x)
    prewitt_y_im = signal.convolve2d(image_true, prewitt_y)
    sobel_x_im = signal.convolve2d(image_true, sobel_x)
    sobel_y_im = signal.convolve2d(image_true, sobel_y)
    
    plt.figure()
    plt.imshow(prewitt_x_im, cmap='gray')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(prewitt_y_im, cmap='gray')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(sobel_x_im, cmap='gray')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(sobel_y_im, cmap = 'gray')
    plt.axis('off')
    
    plt.show()
   
   
    
if __name__ == "__main__":
    run()
    
    
