# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:15:18 2019

@author: Sara Blichowska, Katarzyna Chlebicka
"""


from scipy import ndimage
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from medpy.io import load
import queue
from skimage import measure
import cv2
import sys

# rg_global - global region growing
# rg_local - local region growing
# You may choose which method to use. Local method is used in the written code, but with coordinates that works as global 
#region growing ([-1, 0, 1]). To use modified local method, change the coordinates in line 44

def rg_global(image, seed, margin):
    	
    output_image = np.zeros(image.shape)	
    initial_value = image[seed]	
    uth = initial_value + margin	
    dth = initial_value - margin	
    image_thresholded = np.logical_and(image < uth, image > dth)	
    image_labeled = measure.label(image_thresholded, background=0)	
    output_image[image_labeled == image_labeled[seed]] = 1	
    return output_image

def rg_local(image, seed, margin):
	
    output_image = np.zeros(image.shape)
    y_size, x_size = image.shape
    initial_value = image[seed]
    uth = initial_value + margin
    dth = initial_value - margin

    def get_neighbours(coord):
        output = []  		
        iss = [-1, 0, 1]  		# Here you may change the coordinates
        for xs in iss:	
            for ys in iss:
                c_ys = min(max(coord[0] + ys, 0), y_size - 1)
                c_xs = min(max(coord[1] + xs, 0), x_size - 1)
                output.append((c_ys, c_xs))
        return output


	
    image_queue = queue.Queue()
    image_queue.put(seed)
    output_image[seed] = 1
    our_set = set()
	
    while not image_queue.empty():
        item = image_queue.get()
        neighbours = get_neighbours(item)
        for neighbour in neighbours:	
            if neighbour in our_set:
                continue
            if image[neighbour] < uth and image[neighbour] > dth:
                output_image[neighbour] = 1
                image_queue.put(neighbour)
                our_set.add(neighbour)
    return output_image



def true_false(img):
    voxel = np.sum(img > 0)
    return voxel



def normalize(img):
    img_minimal = np.min(img)
    return (img-img_minimal)/(np.max(img)-img_minimal)

def high_pass(img):
    blurred_image = ndimage.gaussian_filter(img, 2)
    filter_blurred_image = ndimage.gaussian_filter(blurred_image, 0.5)
    alpha = 300
    sharpened = blurred_image + alpha * (blurred_image - filter_blurred_image)
    return sharpened


def run():
    
    print("Which patient do you want to choose?\na)Patient 01, press 1\nb)Patient 02, press 2\nc)Patient 03, press 3\nd)Patient 04, press 4\n")
    select = int(input("Choose the patient to examine: "))

    if select==1:
        path='C:\\Users\\Sara\\Desktop\\tom_project\\patient_1.mha'
        path_2 = 'C:\\Users\\Sara\\Desktop\\tom_project\\1_obrysy.mha'
        pix = (160,220) 
        value = int(input("Select a scan number in range <1;41> : "))
        size = int(35)
        size_2 = int(35)
        distance=0.667969
    elif select==2:
        path='C:\\Users\\Sara\\Desktop\\tom_project\\Patient02.mha'
        path_2 = 'C:\\Users\\Sara\\Desktop\\tom_project\\2_obrys.mha'
        pix = (160,220)
        value = int(input("Select a scan number in range <1;92> : "))
        size = int(91)
        size_2 = int(91)
        distance=0.621094
    elif select==3:
        path='C:\\Users\\Sara\\Desktop\\tom_project\\Patient03.mha'
        path_2 = 'C:\\Users\\Sara\\Desktop\\tom_project\\3_obrysy.mha'
        pix = (160,220)
        value = int(input("Select a scan number in range <1;97> : "))
        size = int(96)
        size_2 = int(96)
        distance=0.683594
    elif select==4:
        path='C:\\Users\\Sara\\Desktop\\tom_project\\Patient04.mha'
        path_2 = 'C:\\Users\\Sara\\Desktop\\tom_project\\4_obrysy.mha'
        pix = (160,220)
        value = int(input("Select a scan number in range <1;41> : "))
        size = int(40)
        size_2 = int(40)
        distance=0.667969
    else:
        print("No such a patient.")
        sys.exit()
        
    
       
    
    
    #loading the picture
    image_data, image_header = load(path)
    image=color.rgb2gray(image_data[:,:,value])
    
    #loading images manually segmented in a program ITK-SNAP
    image_data_2, image_header_2 = load(path_2)
    image_2=color.rgb2gray(image_data_2[:,:,value])
    
    #showing the selected image
    plt.figure()
    plt.imshow(image_data[:,:,value], cmap='gray')
    plt.plot()

    
    #data size
    a = image_data.shape 
    print("Data size: " , a)



    local_image=rg_local(image,pix,18)
    global_image=rg_global(image,pix,18)



    kernel = np.ones((5,5),np.uint8)
  
    #dilation
    dilation = cv2.dilate(local_image,kernel,iterations = 3)
    
    #closing
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    plt.figure()
    plt.imshow(closing, cmap='gray')
    
    #getting voxel distance
    spacing=image_header.get_voxel_spacing()
    print("Voxel distance: ", spacing)
    

    suma = 0 #for our segmented image
    suma_2 = 0 #for image segmented in ITK-SNAP
    
    
    #volume calculations for automatically segmented liver
    for i in range(1, size):
        #loading image
        image=color.rgb2gray(image_data[:,:,i])
        # rozrost obszarow
        local_image=rg_local(image,pix,18)
        global_image=rg_global(image,pix,18) 
        #kernel
        kernel = np.ones((5,5),np.uint8)
        #further segmentation
        dilation = cv2.dilate(local_image,kernel,iterations = 3)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        spacing=image_header.get_voxel_spacing()
        a = true_false(closing)
        suma = suma+ a

  # volume calculations for manually segmented liver      
    for i in range(1, size_2):
        image_2=color.rgb2gray(image_data_2[:,:,i])
        b = true_false(image_2)
        suma_2 = suma_2+b
        
        
        
    
    print("Number of pixels: ", suma)  
    print("Number of pixels of manually segmented images: ", suma_2)
    volume = suma * 5.0 / 1000 * distance**2
    volume_2 = suma_2 * 5.0 / 1000 * distance**2
    print("LIVER VOLUME: ", volume, "  CM^3")
    print("LIVER VOLUME ITK-SNAP: ", volume_2, "CM^3")
    
    
    size_done = image_data.shape
    print(size_done)
        
    
    
if __name__ == "__main__":
    run()
