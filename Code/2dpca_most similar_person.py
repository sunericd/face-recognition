#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:59:29 2018

@author: wassy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import random


#for per_section in range(1,20):
path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/'



def run_2dpca(components,num_classes,per_section,dirs,test_dirs ,path_dir):
    leaf_mats = []
    leaf_unchanged = []

    pic_list = []
    test_paths = []
    for k in range(0,len(dirs)):
        temp = os.listdir(path_dir + dirs[k])
        for ww in range(0,len(temp)):
            pic_list.append(dirs[k] +temp[ww])
            
    for k in range(0,len(test_dirs)):
        print(test_dirs)
        temp = os.listdir(path_dir + test_dirs[k])
        test_paths.append(test_dirs[k] +temp[0])
    print(test_paths)

    #this was for the trying to see which photo an image outside of the test set looks like
    #test_paths.append('9338527/9338527.8.jpg')
        
    train_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(pic_list)/num_classes)) for x in dirs))
    test_class = list(itertools.chain.from_iterable(itertools.repeat(x, len(test_paths)) for x in test_dirs))
    #test_class.append('9338527/')
    for f in pic_list:
        im = plt.imread(path_dir+f)
        im.setflags(write=1)
        leaf_unchanged.append(im)
        # read image as 3D numpy array
        S = np.subtract(1,im) # S = 1-p
        # Make a col vector of the image [row by row] [R --> G --> B]
        gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))
        leaf_mats.append(gr)
    test_set = []  
    test_unchanged = []
    for f in test_paths:
        im = plt.imread(path_dir+f)
        test_unchanged.append(im)
        # read image as 3D numpy array
        S = np.subtract(1,im) # S = 1-p
        # Make a col vector of the image [row by row] [R --> G --> B]  
        gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))
        test_set.append(gr)    
    L = len(leaf_mats)
    sum_mat = 0 
    for i in range(0,len(leaf_mats)):
        sum_mat = sum_mat + leaf_mats[i]
    S_avg =  sum_mat/L   
    #calculate the G
    G = 0 
    for k in range(0,len(leaf_mats)): 
        G =  np.matmul( np.transpose(leaf_mats[k] - S_avg )  ,leaf_mats[k] - S_avg ) 
    G = G / L
    # we want to find the largest eigenvector of G
    eva = np.linalg.eig(G)[0]
    evec = np.linalg.eig(G)[1]    
    max_places = eva.argsort()[-components:][::-1]
    max_vector = evec[:, max_places]
    return(max_vector, train_class, test_class, leaf_mats, test_set, leaf_unchanged, test_unchanged, S_avg)

    
def make_feature_matrix(sample_image,max_vector, d):
    feature_matrix = []
    for i in range(0,d):
        sample_image.astype(float)        
        feature_matrix.append(np.matmul(sample_image,max_vector[:,i]))
    return(np.array(feature_matrix))

def distance_function(samp1, samp2,d):
    total_distance = 0 
    feat1 = make_feature_matrix( samp1 , max_vector,d)
    feat2 = make_feature_matrix( samp2 , max_vector,d )
    for l in range(0,d):
        total_distance = total_distance + np.linalg.norm(np.array(feat1[l]) - np.array(feat2[l]))
    return(total_distance)

def score(test_set, leaf_mats, train_class,test_class, components,leaf_unchanged,test_unchanged ):
    guess_class = []
    correct = []
    for k in range(0,len(test_set)):        
        distance_to_each = []
        for i in range(0,len(leaf_mats)):
            distance_to_each.append(distance_function(test_set[k], leaf_mats[i] ,components))
        min(distance_to_each)
        min_ind = distance_to_each.index(   min(distance_to_each))
        print("test image below")
        plt.imshow(test_unchanged[k])
        plt.show()
        #print("test image" + str(k) )
        guess_class.append(train_class[min_ind])
        print("most similar below")
        plt.imshow(leaf_unchanged[min_ind])
        plt.show()
        print(len(train_class))
        print(min_ind)
        print(len(test_class))
        print(k)
        if train_class[min_ind] == test_class[k]:
            correct.append(1)
        else:
            correct.append(0)
    return(sum(correct)/ len(correct))

def graph_results(x_values, y_values, file_name, color, title, xmin, xmax):
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, color[0])
    ax.fill_between(x_values, 0, y_values, facecolor=color, alpha=0.2)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_xlabel(title, fontsize=12)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0,1.05)
    plt.tight_layout()
    plt.savefig('/Users/wassy/Documents/fall_2018/am205/final_project/github/wassy_pictures/' + file_name, dpi=500)


#most similar
dirs = ["9326871/", "9332898/", "9338446_Star/", "9338454/", "9338462/", "9338489/", "9338497/", "9338519/", "9338527/", "9338543/","9414649/", "9416994/"]
for dd in range(len(dirs)):
    dirs = ["9326871/", "9332898/", "9338446_Star/", "9338454/", "9338462/", "9338489/", "9338497/", "9338519/", "9338527/", "9338543/","9414649/", "9416994/"]
    test_dirs = [dirs.pop(dd)]
    max_vector, train_class, test_class, leaf_mats, test_set, leaf_unchanged, test_unchanged,avg_face =  run_2dpca( 3, 11, 0, dirs,test_dirs, path_dir )
    print( score(test_set, leaf_mats, train_class,test_class, 3, leaf_unchanged, test_unchanged ) )





#avg_for_show= np.empty((200,180,3))
#for i in range(200):
#    for j in range(180):
#        for k in range(3):
#            avg_for_show[i,j,k] = avg_face[ i+k *3 ,j ]
#
#plt.imshow(np.subtract(1,avg_for_show))
#
