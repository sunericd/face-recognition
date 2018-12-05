#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:59:29 2018

@author: wassy
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Reading and converting to subtractive representation
path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/9338446/'
leaf_mats = []

for f in os.listdir(path_dir):
    im = plt.imread(path_dir+f) # read image as 3D numpy array
    S = np.subtract(1,im) # S = 1-p
    # Make a col vector of the image [row by row] [R --> G --> B]
    
    gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))

    leaf_mats.append(gr)

    
# Calculate the average 
L = len(leaf_mats)

sum_mat = 0 
for i in range(len(leaf_mats)):
    sum_mat = sum_mat + leaf_mats[i]
S_avg =  sum_mat/L   

#calculate the G
G = 0 
for k in range(len(leaf_mats)): 
    G =  np.matmul( np.transpose(leaf_mats[k] - S_avg )  ,leaf_mats[k] - S_avg ) 

G = G / 20

# we want to find the largest eigenvector of G

eva = np.linalg.eig(G)[0]
evec = np.linalg.eig(G)[1]
    
max_e_val = max(eva)
    
first = np.where( eva == max_e_val)[0][0]

max_vector = evec[:, 0]

# Reshape and plot S_avg as a picture
m = 200
n = 180
d = 3
S_reformatted = S_avg.reshape((m,n,d)) # recast to 3D matrix
S_reformatted_rgb = np.subtract(1,S_reformatted) # recast to RGB

plt.figure()
plt.imshow(S_reformatted_rgb)
plt.show()