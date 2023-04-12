
#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Telmo Subirá Rodríguez   
# Created Date: 2023/01/21
# version ='1.0'
# ---------------------------------------------------------------------------
# This file is part of the Master's Thesis research work of the author
# on the Valencia International University (VIU).
# The file contains auxiliary functions defined to ease the 
# hyperspectral images management from a local directory.

__author__ = "Telmo Subirá Rodríguez"
__version__ = "1.0.0"
__email__ = "telmosubirar@gmail.com"
__status__ = "Prototype"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from spectral import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
from skimage.measure import label
from sklearn.decomposition import PCA
# import torch

# -----------------------------------

# Create a pandas DataFrame with the name of the files in the data folder
def create_df_from_files_in_path(files_path, verbose = False):
    filenames = []
    if verbose: print(f'Reading file names from {files_path}...')
    for filename in os.listdir(files_path):
        filenames.append(filename)
    files = pd.Series(filenames).str.split('.')
    format = (files.str[1]).str.strip()
    names = (files.str[0]).str.strip()

    df_files = pd.DataFrame(list(zip(names, format, filenames)),
                columns =['Filename', 'Format', 'Path'])

    heights = []
    widths = []
    layers = []

    for id in df_files.index:
        anomaly_map, hs_image = load_HSI_from_idx(id, df_files)
        shape = np.shape(hs_image)
        heights.append(shape[0])
        widths.append(shape[1])
        layers.append(shape[2])

    df_files['Height'] = heights
    df_files['Width'] = widths
    df_files['Layers'] = layers

    if verbose: print(df_files)

    return df_files

# -----------------------------------

# Load a HS image from the folder and extract the image and the anomaly map.
def load_HSI_from_idx(id_img, df, verbose = False):
    path_img = os.path.join("data", df.Path[id_img])
    if verbose: print(f'Reading HSI image from {path_img}...')
    hs_image_structure = loadmat(path_img)
    anomaly_map = hs_image_structure['map']
    hs_image = hs_image_structure['data']
    if verbose: 
        height, width, freq = np.shape(hs_image)
        height_map, width_map = np.shape(anomaly_map)
        print(f'Hypercube {df.Filename[id_img]} data structure generated with dimensions {height}x{width}x{freq}.')
        print(f'Anomaly map data structure generated with dimensions {height_map}x{width_map}.')

    return anomaly_map, hs_image

# -----------------------------------

# Load a HS image from the folder and extract the image and the anomaly map.
def load_HSI_from_path(path_img, verbose = False):
    if verbose: print('Reading HSI image from {path_img}...')
    hs_image_structure = loadmat(path_img)
    anomaly_map = hs_image_structure['map']
    hs_image = hs_image_structure['data']
    if verbose: 
        height, width, freq = np.shape(hs_image)
        height_map, width_map = np.shape(anomaly_map)
        print(f'Hypercube {path_img} data structure generated with dimensions {height}x{width}x{freq}.')
        print(f'Anomaly map data structure generated with dimensions {height_map}x{width_map}.')

    return anomaly_map, hs_image

# -----------------------------------

# Print an RGB-like image of the hypercube, using three spectral layers
def print_RGB_HSI(hs_image, img_name = None, layers = (29, 19, 9)):
    view = imshow(hs_image, layers)
    if img_name != None: plt.title(f'RGB-like view of {img_name}')
    plt.axis('off')
    plt.show()

# -----------------------------------

# Print an overlay of the anomaly mask over the RGB-like image of the hypercube
def print_anomaly_map_over_RGB_HSI(hs_image, anomaly_map, img_name = None, layers = (29, 19, 9)):
    view = imshow(hs_image, layers, classes=anomaly_map)
    view.set_display_mode('overlay')
    view.class_alpha = 0.5
    if img_name != None: plt.title(f'Anomaly mask overlay (red) for {img_name}')
    plt.axis('off')
    plt.show()

# -----------------------------------

# Print an RGB-like image of the hypercube, using three spectral layers
def print_RGB_HSI_with_mask(hs_image, mask, cmap = plt.cm.hot, img_name = None, alpha = 0.7, layers = (29, 19, 9), title = ''):
    fig = plt.figure
    im1 = imshow(hs_image, layers)
    im2 = plt.imshow(mask, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    if img_name != None: plt.title(f'RGB-like {img_name} + attention mask')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function that plots the hypercube as a 3D scatter plot.
# NOTE: for using with big data structures. It may take long time to plot the image.

def scatter_hypercube_from_hyperspectral_image(hs_image, save_img = False, img_name = 'hs_image.jpg'):
    height, width, num_bands = hs_image.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for layer in range(num_bands):
        # Identify the amplitude values of the band
        band = hs_image[:,:,layer]
        # Create a 2D mesh for the band, with X-Y dims
        x, y = np.linspace(0, width-1, width), np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)
        # Plot the bands from top to bottom. The color of every band will be determined by its amplitude.
        ax.scatter(X, Y, num_bands-layer, c = band, cmap='jet', s = 1)
    plt.axis('off')
    plt.show()
    if save_img:
        fig.savefig(img_name)

# -----------------------------------

# Function that displays the hypercube as a stack of all the 2D layers.
# NOTE: for using with small data structures. It may take long time to plot the image.
def surf_hypercube_from_hyperspectral_image(data, save_img = False, img_name = 'hs_image.jpg'):
    height, width, num_bands = data.shape
    X = np.arange(width)
    Y = np.arange(height)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_axis_off()
    min = np.min(data)
    max = np.max(data)
    for layer in np.arange(num_bands):
        img = data[:,:,layer]
        img -= np.min(img)
        img = img/np.max(img)
        ax.plot_surface(X, Y, num_bands-layer+Z, rstride=1, cstride=1, facecolors = cm.jet(img))
    plt.show()
    if save_img:
        fig.savefig(img_name)

# -----------------------------------

# Function that displays a matrix of patches extracted from a HSI,
# every patch as a stack of all the 2D layers.
def surf_hypercube_patches(patches, save_img = False, img_name = 'hs_patches_image.jpg'):
    num_patches_height, num_patches_width, height, width, num_bands = np.shape(patches)
    fig, axs = plt.subplots(num_patches_height, num_patches_width, subplot_kw={"projection": "3d"})
    for h in np.arange(num_patches_height):
        for w in np.arange(num_patches_width):
            data = patches[h,w,:]
            axs[h,w].set_axis_off()
            X = np.arange(width)
            Y = np.arange(height)
            X, Y = np.meshgrid(X, Y)
            Z = np.zeros_like(X)
            for layer in np.arange(num_bands):
                img = data[:,:,layer]
                img -= np.min(img)
                img = img/np.max(img)
                axs[h,w].plot_surface(X, Y, num_bands-layer+Z, rstride=1, cstride=1, facecolors = cm.jet(img))
    if save_img:
        fig.savefig(img_name)

# -----------------------------------

# Function that displays the a matrix of patches extracted from an anomaly map,
def plot_anomaly_map_patches(patches, save_img = False, img_name = 'map_patches_image.jpg'):
    num_patches_height, num_patches_width, height, width = np.shape(patches)
    fig, axs = plt.subplots(num_patches_height, num_patches_width)
    for h in np.arange(num_patches_height):
        for w in np.arange(num_patches_width):
            axs[h,w].set_axis_off()
            axs[h,w].imshow(patches[h,w], cmap='gray')   
    if save_img:
        fig.savefig(img_name)

# -----------------------------------

# Function that divides the input HSI into cube-shaped patches. Optionally divides the anomaly map in the same proportion.
def slice_HSI_into_cube_patches(hs_image, anomaly_map = None, slice_map = False, patch_dim = 10, overlap = 0, normalize = True, verbose = False):
    # Get the dimensions of the HSI
    height, width, num_bands = np.shape(hs_image)
    if verbose:
        print(f'HSI with dimensions:\nHeight = {height}\nWidth = {width}\nSpectral layers = {num_bands}')
    total_pixels_HSI = height*width
    if normalize:
        hs_image = (hs_image/hs_image.max())
    # Get the size of the patches and the number of patches
    num_patches_height = int((height-overlap) / (patch_dim-overlap))
    num_patches_width = int((width-overlap) / (patch_dim-overlap))
    if verbose:
        print(f'Spatial dimension of patches = {patch_dim}\nNumber of patches in the height dimension = {num_patches_height}\nNumber of patches in the width dimension = {num_patches_width}')
   
    # Initialize with 0's the structure containing the patches
    HSI_patches = np.zeros((num_patches_height, num_patches_width, patch_dim, patch_dim, num_bands))
    # Slice the HSI and store the patches in the structure
    for h in np.arange(num_patches_height):
        for w in np.arange(num_patches_width):
            ini_row = patch_dim*h - overlap*h 
            end_row = ini_row + patch_dim
            ini_col = patch_dim*w - overlap*w
            end_col = ini_col + patch_dim
            HSI_patches[h,w] = hs_image[ini_row:end_row, ini_col:end_col,:]
    
    # If the anomaly map was included as an attribute, slice it
    if slice_map == True:
        map_patches = np.zeros((num_patches_height, num_patches_width, patch_dim, patch_dim))
        # Slice the map and store the patches in the structure
        for h in np.arange(num_patches_height):
            for w in np.arange(num_patches_width):
                ini_row = patch_dim*h - overlap*h 
                end_row = ini_row + patch_dim
                ini_col = patch_dim*w - overlap*w
                end_col = ini_col + patch_dim
                map_patches[h,w] = anomaly_map[ini_row:end_row, ini_col:end_col]

        total_pixels_patches = end_row*end_col
        if verbose:
            print(f'Total pixels in the original image = {total_pixels_HSI}\nTotal pixels in the patches = {total_pixels_patches}')
        if total_pixels_HSI != total_pixels_patches:
            print(f'>> WARNING! Some information will be lost! \nIt is not possible to make use of all the image data with this slicing proportion. \nThe original image has {total_pixels_HSI} hyperspectral pixels, while the patches contain a total of {total_pixels_patches} pixels together.')

        # Return both the HSI patches and the map patches
        return HSI_patches, map_patches
    
    # If there is not anomaly map, just return the patches of the HSI
    return HSI_patches

# -----------------------------------

# Function that generates a HSI structure with random data for testing.
def generate_random_HSI(height, width, layers):
    return np.linspace(0,(height*width*layers)-1,height*width*layers).reshape([height,width,layers])

# -----------------------------------

# Function that saves the patches of every image listed in a DataFrame into files, via pickle
def save_patches_to_pickle(list_HSI_patches, list_HSI_maps, df, path = "data_patches"):
    for id_img in df.index:
        patched_file_path = os.path.join(path, df.Filename[id_img])
        with open(patched_file_path, 'wb') as fp:
            pickle.dump(list_HSI_patches[id_img], fp)
        patched_map_path = os.path.join("data_patches", df.Filename[id_img]+"_anomaly_map")
        with open(patched_map_path, 'wb') as fp:   
            pickle.dump(list_HSI_maps[id_img], fp)

# -----------------------------------

# Function that reads the patches of every image listed in a DataFrame from files, via pickle
def read_patches_from_pickle(df, path = "data_patches"):
    ABU_sliced_imgs_read = []
    ABU_sliced_maps_read = []
    # Reading all the patches of the images in the ABU dataset
    for id_img in df.index:
        # Load HSI
        patched_file_path = os.path.join(path, df.Filename[id_img])
        with open(patched_file_path, 'rb') as fp:
            ABU_sliced_imgs_read.append(pickle.load(fp))
        patched_map_path = os.path.join(path, df.Filename[id_img]+"_anomaly_map")
        with open(patched_map_path, 'rb') as fp:   
            ABU_sliced_maps_read.append(pickle.load(fp))

    return ABU_sliced_imgs_read, ABU_sliced_maps_read

# -----------------------------------

# def reconstruct_HSI_tensor_to_numpy(hs_img):
#     hs_img = torch.permute(hs_img, (0,2,3,1))
#     hs_img = hs_img.cpu().detach().numpy()
#     hs_img = np.squeeze(hs_img, axis=0)
#     return hs_img

# -----------------------------------
def plot_learning_curves(path_df, entropy = True):
    training_df = pd.read_csv(path_df)

    fig, axes = plt.subplots(4, 1, figsize=(6,12))
    plt.subplot(411)
    plt.plot(training_df.KL_div_losses, color='r', label='KL')
    plt.subplot(412)
    plt.plot(training_df.reconstruction_losses, color='g', label='RL')
    plt.subplot(413)
    plt.plot(training_df.spectral_angle_losses, color='k', label='SAD')
    if entropy == True:
        plt.subplot(414)
        plt.plot(training_df.entropy_losses, color='b', label='H')
    
    for ax in axes:
        ax.legend()

# -----------------------------------
def norm_img(img):
    img = img - np.min(img)
    img = img/np.max(img)
    return img

# -----------------------------------
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# -----------------------------------
def filter_image_under_threshold(image, thr, value = 0):
    image[image<thr] = value
    return image

# -----------------------------------
def filter_image_over_threshold(image, thr, value = 1):
    image[image>thr] = value
    return image

# -----------------------------------
def compute_pd(ground_truth, anomaly_map):
    # Number of pixels different than 0 when the mask pixels are different than 0
    num_pixels = np.sum((anomaly_map != 0) & (ground_truth != 0))
    # Calculate the probability
    total_pixels = np.sum(ground_truth != 0)
    return num_pixels / total_pixels

# -----------------------------------
def compute_prob_localization(ground_truth, anomaly_map):
    # Label the connected components in the reference image
    labels = label(ground_truth)
    # Count the number of connected components
    num_objects = labels.max()
    # Initialize a list to store the number of objects that meet the criteria
    num_meet_criteria = []
    # Loop through each connected component
    for i in range(1, num_objects+1):
        # Extract the connected component
        obj = (labels == i)
        # Check if there is at least one non-zero pixel in the corresponding
        # area of the other image
        if np.any(obj & (anomaly_map != 0)):
            num_meet_criteria.append(1)
        else:
            num_meet_criteria.append(0)
    # Calculate the probability
    probability = np.sum(num_meet_criteria) / num_objects
    return probability


# -----------------------------------
def compute_pfa(ground_truth, anomaly_map):
    # Number of pixels different than 0 when the mask pixels are different than 1
    num_pixels = np.sum((anomaly_map != 0) & (ground_truth != 1))
    # Calculate the probability
    total_pixels = np.sum(ground_truth != 1)
    return num_pixels / total_pixels

# -----------------------------------
def compute_ROC(ground_truth, anomaly_map, vmin = 0, vmax = 1, steps = 0.05):
    if ground_truth.shape != anomaly_map.shape:
        print(">> ERROR! The dimensions of the two images are not the same.")
        return
    pd_list = []
    pfa_list = []
    pl_list = []
    for threshold in np.arange(vmin, vmax+steps, steps):
        thresholded_map = filter_image_under_threshold(anomaly_map, threshold)
        pd_list.append(compute_pd(ground_truth, thresholded_map))
        pl_list.append(compute_prob_localization(ground_truth, thresholded_map))
        pfa_list.append(compute_pfa(ground_truth, thresholded_map))
    return pd_list, pl_list, pfa_list

# -----------------------------------
def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

# -----------------------------------
def SAD(img, img_hat):
    img_SAD = np.zeros_like(img[:,:,0])
    if np.shape(img) != np.shape(img_hat):
        print("Error. Input image dimmensions don't match.")
        return img_SAD

    # Compute the SAD for every pixel in the image
    for row in np.arange(np.shape(img)[0]):
        for col in np.arange(np.shape(img)[1]):
            pixel = img[row, col, :]
            pixel_hat = img_hat[row, col, :]
            theta = np.dot(pixel,pixel_hat) / (np.linalg.norm(pixel) * np.linalg.norm(pixel_hat))
            img_SAD[row, col] = np.arccos(theta)

    # Return the matrix with SAD values
    return img_SAD

import cv2
# -----------------------------------
def CFAR_2D(img, thr_type = 'higher', filter_dims = 4, gap_pixels = 1, thr_factor = 0.15):
    height, width = np.shape(img)
    anomaly_map = np.zeros_like(img)
    # Add padding to the image
    img = cv2.copyMakeBorder(img, 
                             filter_dims, 
                             filter_dims, 
                             filter_dims, 
                             filter_dims, 
                             cv2.BORDER_REPLICATE)
    
    # Loop over the pixels and check if the pixel under test surpasses the threshold
    for row in range(filter_dims, height+filter_dims):
        for col in range(filter_dims, width+filter_dims):
            # Get the value of the pixel and the average of the surroundings, discounting the gap pixels
            value_under_test = img[row, col]
            patch_gap = img[row-gap_pixels:row+gap_pixels, col-gap_pixels:col+gap_pixels]
            patch_under_test = img[row-filter_dims:row+filter_dims, col-filter_dims:col+filter_dims]
            sum_patch = np.sum(patch_under_test) - np.sum(patch_gap)
            average_level = sum_patch / (np.size(patch_under_test)-np.size(patch_gap))
            # Check the threshold condition
            if thr_type == 'higher':
                if value_under_test > average_level*thr_factor:
                    anomaly_map[row-filter_dims, col-filter_dims] = value_under_test
                else:
                    anomaly_map[row-filter_dims, col-filter_dims] = 0

            elif thr_type == 'lower':
                if value_under_test < average_level*thr_factor:
                    anomaly_map[row-filter_dims, col-filter_dims] = 1-value_under_test
                else:
                    anomaly_map[row-filter_dims, col-filter_dims] = 0  
    return anomaly_map

# ----------------------------------- 
def reduce_HSI_dim_with_PCA(hs_image, PCA_layers):
    dims = np.shape(hs_image)
    q = hs_image.reshape(-1, hs_image.shape[2])
    pca = PCA(n_components = PCA_layers)
    return pca.fit_transform(q).reshape(dims[0], dims[1], PCA_layers)

# ----------------------------------- 
def plot_HSI_layers(hs_image, PCA_layers):
    fig = plt.figure(figsize = (20, 10))
    for i in range(1, 1+PCA_layers):
        fig.add_subplot(2,4, i)
        plt.imshow(hs_image[:, :, i-1], cmap='hot')
        plt.axis('off')
        plt.title(f'PCA-HSI layer {i}')
     