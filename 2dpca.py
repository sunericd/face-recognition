#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:59:29 2018

@author: wassy
"""

import numpy as np
import matplotlib.pyplot as plt
import os

path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/9338446/'
leaf_mats = []
leaf_unchanged = []

for f in os.listdir(path_dir)[:len(os.listdir(path_dir)) - 1]:
    im = plt.imread(path_dir+f) # read image as 3D numpy array
    leaf_unchanged.append(im)
    S = np.subtract(1,im) # S = 1-p
    # Make a col vector of the image [row by row] [R --> G --> B]
    
    gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))

    leaf_mats.append(gr)
    
test_set = []
test_unchanged = []
for f in os.listdir(path_dir)[len(os.listdir(path_dir)) - 1:]:
    im = plt.imread(path_dir+f) # read image as 3D numpy array
    S = np.subtract(1,im) # S = 1-p
    # Make a col vector of the image [row by row] [R --> G --> B]
    
    gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))
    test_unchanged.append(im)
    test_set.append(gr)

    
L = len(leaf_mats)

sum_mat = 0 
for i in range(len(leaf_mats)):
    sum_mat = sum_mat + leaf_mats[i]
S_avg =  sum_mat/L   

#calculate the G
G = 0 
for k in range(len(leaf_mats)): 
    G =  np.matmul( np.transpose(leaf_mats[k] - S_avg )  ,leaf_mats[k] - S_avg ) 

G = G / L

# we want to find the largest eigenvector of G

eva = np.linalg.eig(G)[0]
evec = np.linalg.eig(G)[1]
    
k = 8
max_places = eva.argsort()[-k:][::-1]
    


max_vector = evec[:, max_places]

def make_feature_matrix(sample_image,max_vector, d):
    feature_matrix = []
    for i in range(d):
        feature_matrix.append(np.matmul(sample_image,max_vector[:,i]))
    return(feature_matrix)

def distance_function(samp1, samp2,d):
    total_distance = 0 
    for l in range(d):
        feat1 = make_feature_matrix( samp1 , max_vector,l )
        feat2 = make_feature_matrix( samp2 , max_vector,l )
        total_distance = total_distance + np.linalg.norm(np.array(feat1) - np.array(feat2))
    return(total_distance)

for k in range(len(test_set)):
    test = make_feature_matrix(test_set[k], max_vector, 8)
    
    distance_to_each = []
    for i in range(len(leaf_mats)):
        distance_to_each.append(distance_function(test_set[k], leaf_mats[i] ,8 ))
    
    
    min(distance_to_each)
    min_ind = distance_to_each.index(   min(distance_to_each))
    plt.imshow(leaf_unchanged[min_ind])
    plt.show()
    plt.imshow(test_unchanged[k])

# Reshape and plot S_avg as a picture
#m = 200
#n = 180
#d = 3
#S_reformatted = S_avg.reshape((m,n,d)) # recast to 3D matrix
#S_reformatted_rgb = np.subtract(1,S_reformatted) # recast to RGB
#
#plt.figure()
#plt.imshow(S_reformatted_rgb)
#plt.show()