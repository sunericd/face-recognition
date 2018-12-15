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

accuracy_per_section= []
for per_section in range(1,20):
    
    components = 3

    dirs = ["9326871/", "9332898/", "9338446_Star/", "9338454/", "9338462/", "9338489/", "9338497/", "9338519/", "9338527/", "9338543/","9414649/", "9416994/"]
    num_classes = len(dirs)
    path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/'
    leaf_mats = []
    leaf_unchanged = []
    
    pic_list = []
    test_paths = []
    for k in range(len(dirs)):
        temp = os.listdir(path_dir + dirs[k])
        random.shuffle(temp)
        for j in range(per_section):
            test_paths.append(dirs[k] +temp[j])
        for ww in range(per_section,len(temp)):
            pic_list.append(dirs[k] +temp[ww])
            
    #test_paths.append('9338527/9338527.8.jpg')
        
    
    train_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(pic_list)/num_classes)) for x in dirs))
    test_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(test_paths)/num_classes)) for x in dirs))
    #test_class.append('9338527/')
    
    
    for f in pic_list:
        im = plt.imread(path_dir+f) # read image as 3D numpy array
        leaf_unchanged.append(im)
        S = np.subtract(1,im) # S = 1-p
        # Make a col vector of the image [row by row] [R --> G --> B]
        
        gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))
    
        leaf_mats.append(gr)
        
    test_set = []  
    test_unchanged = []
    for f in test_paths:
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
        
    max_places = eva.argsort()[-components:][::-1]
        

    
    max_vector = evec[:, max_places]
    
    def make_feature_matrix(sample_image,max_vector, d):
        feature_matrix = []
        for i in range(d):
            sample_image.astype(float)        
            feature_matrix.append(np.matmul(sample_image,max_vector[:,i]))
        return(np.array(feature_matrix))
    
    def distance_function(samp1, samp2,d):
        total_distance = 0 
        feat1 = make_feature_matrix( samp1 , max_vector,d)
        feat2 = make_feature_matrix( samp2 , max_vector,d )
        for l in range(d):
            total_distance = total_distance + np.linalg.norm(np.array(feat1[l]) - np.array(feat2[l]))
        return(total_distance)
    
    guess_class = []
    correct = []
    for k in range(len(test_set)):
        test = make_feature_matrix(test_set[k], max_vector, components)
        
        distance_to_each = []
        for i in range(len(leaf_mats)):
            distance_to_each.append(distance_function(test_set[k], leaf_mats[i] ,components))
        
        
        min(distance_to_each)
        min_ind = distance_to_each.index(   min(distance_to_each))
        #print("nearest neighbor:" + str(k) )
        #plt.imshow(leaf_unchanged[min_ind])
        plt.show()
        #print("test image" + str(k) )
        #plt.imshow(test_unchanged[k])
        guess_class.append(train_class[min_ind])
        #print(pic_list[min_ind])
        if train_class[min_ind] == test_class[k]:
            correct.append(1)
        else:
            correct.append(0)
    
    accuracy_per_section.append( sum(correct)/ len(correct))
    print(per_section)
plt.plot(range(1,20), accuracy_per_section)

fig, ax = plt.subplots()
ax.plot(range(1,20), accuracy_per_section, 'r')
ax.fill_between(range(1,20), 0, accuracy_per_section, facecolor='red', alpha=0.2)
ax.set_ylabel('Classification Accuracy', fontsize=12)
ax.set_xlabel('Test Set Size', fontsize=12)
ax.set_xlim(2,18)
ax.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig('/Users/wassy/Documents/fall_2018/am205/final_project/github/wassy_pictures/test_size_2dpca', dpi=500)


accuracy_per_section= []
for components in range(1,13):
    per_section = 4
    
    
    dirs = ["9326871/", "9332898/", "9338446_Star/", "9338454/", "9338462/", "9338489/", "9338497/", "9338519/", "9338527/", "9338543/","9414649/", "9416994/"]
    num_classes = len(dirs)
    path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/'
    leaf_mats = []
    leaf_unchanged = []
    
    pic_list = []
    test_paths = []
    for k in range(len(dirs)):
        temp = os.listdir(path_dir + dirs[k])
        random.shuffle(temp)
        for j in range(per_section):
            test_paths.append(dirs[k] +temp[j])
        for ww in range(per_section,len(temp)):
            pic_list.append(dirs[k] +temp[ww])
            
    #test_paths.append('9338527/9338527.8.jpg')
        
    
    train_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(pic_list)/num_classes)) for x in dirs))
    test_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(test_paths)/num_classes)) for x in dirs))
    #test_class.append('9338527/')
    
    
    for f in pic_list:
        im = plt.imread(path_dir+f) # read image as 3D numpy array
        leaf_unchanged.append(im)
        S = np.subtract(1,im) # S = 1-p
        # Make a col vector of the image [row by row] [R --> G --> B]
        
        gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))
    
        leaf_mats.append(gr)
        
    test_set = []  
    test_unchanged = []
    for f in test_paths:
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
        
    max_places = eva.argsort()[-components:][::-1]
        
    
    
    max_vector = evec[:, max_places]
    
    def make_feature_matrix(sample_image,max_vector, d):
        feature_matrix = []
        for hhh in range(d):
            sample_image.astype(float)        
            feature_matrix.append(np.matmul(sample_image,max_vector[:,hhh]))
        return(np.array(feature_matrix))
    
    def distance_function(samp1, samp2,d):
        total_distance = 0 
        feat1 = make_feature_matrix( samp1 , max_vector,d)
        feat2 = make_feature_matrix( samp2 , max_vector,d )
        for l in range(d):
            total_distance = total_distance + np.linalg.norm(np.array(feat1[l]) - np.array(feat2[l]))
        return(total_distance)
    
    guess_class = []
    correct = []
    for k in range(len(test_set)):
        test = make_feature_matrix(test_set[k], max_vector, components)
        
        distance_to_each = []
        for i in range(len(leaf_mats)):
            distance_to_each.append(distance_function(test_set[k], leaf_mats[i] ,components))
        
        
        min(distance_to_each)
        min_ind = distance_to_each.index(   min(distance_to_each))
        #print("nearest neighbor:" + str(k) )
        #plt.imshow(leaf_unchanged[min_ind])
        #plt.show()
        #print("test image" + str(k) )
        #plt.imshow(test_unchanged[k])
        guess_class.append(train_class[min_ind])
        #print(pic_list[min_ind])
        if train_class[min_ind] == test_class[k]:
            correct.append(1)
        else:
            correct.append(0)
    
    accuracy_per_section.append( sum(correct)/ len(correct))
    print(per_section)

fig, ax = plt.subplots()
ax.plot(range(1,13), accuracy_per_section, 'b')
ax.fill_between(range(1,13), 0, accuracy_per_section, facecolor='blue', alpha=0.2)
ax.set_ylabel('Classification Accuracy', fontsize=12)
ax.set_xlabel('Number of projection vectors, $p$', fontsize=12)
ax.set_xlim(1,13)
ax.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig('/Users/wassy/Documents/fall_2018/am205/final_project/github/wassy_pictures/proj_vectors_2dpca', dpi=500)

accuracy_per_section= []
for num_classes in range(1,13):
    components = 3
    per_section = 4
    
    
    dirs = ["9326871/", "9332898/", "9338446_Star/", "9338454/", "9338462/", "9338489/", "9338497/", "9338519/", "9338527/", "9338543/","9414649/", "9416994/"]
    dirs = dirs[:num_classes]
    path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/'
    leaf_mats = []
    leaf_unchanged = []
    
    pic_list = []
    test_paths = []
    for k in range(len(dirs)):
        temp = os.listdir(path_dir + dirs[k])
        random.shuffle(temp)
        for j in range(per_section):
            test_paths.append(dirs[k] +temp[j])
        for ww in range(per_section,len(temp)):
            pic_list.append(dirs[k] +temp[ww])
            
    #test_paths.append('9338527/9338527.8.jpg')
        
    
    train_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(pic_list)/num_classes)) for x in dirs))
    test_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(test_paths)/num_classes)) for x in dirs))
    #test_class.append('9338527/')
    
    
    for f in pic_list:
        im = plt.imread(path_dir+f) # read image as 3D numpy array
        leaf_unchanged.append(im)
        S = np.subtract(1,im) # S = 1-p
        # Make a col vector of the image [row by row] [R --> G --> B]
        
        gr = np.concatenate((S.transpose(2,0,1)[0],S.transpose(2,0,1)[1],S.transpose(2,0,1)[2] ))
    
        leaf_mats.append(gr)
        
    test_set = []  
    test_unchanged = []
    for f in test_paths:
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
        
    max_places = eva.argsort()[-components:][::-1]
        
    
    
    max_vector = evec[:, max_places]
    
    def make_feature_matrix(sample_image,max_vector, d):
        feature_matrix = []
        for i in range(d):
            sample_image.astype(float)        
            feature_matrix.append(np.matmul(sample_image,max_vector[:,i]))
        return(np.array(feature_matrix))
    
    def distance_function(samp1, samp2,d):
        total_distance = 0 
        feat1 = make_feature_matrix( samp1 , max_vector,d)
        feat2 = make_feature_matrix( samp2 , max_vector,d )
        for l in range(d):
            total_distance = total_distance + np.linalg.norm(np.array(feat1[l]) - np.array(feat2[l]))
        return(total_distance)
    
    guess_class = []
    correct = []
    for k in range(len(test_set)):
        test = make_feature_matrix(test_set[k], max_vector, components)
        
        distance_to_each = []
        for i in range(len(leaf_mats)):
            distance_to_each.append(distance_function(test_set[k], leaf_mats[i] ,components))
        
        
        min(distance_to_each)
        min_ind = distance_to_each.index(   min(distance_to_each))
        #print("nearest neighbor:" + str(k) )
        #plt.imshow(leaf_unchanged[min_ind])
        plt.show()
        #print("test image" + str(k) )
        #plt.imshow(test_unchanged[k])
        guess_class.append(train_class[min_ind])
        #print(pic_list[min_ind])
        if train_class[min_ind] == test_class[k]:
            correct.append(1)
        else:
            correct.append(0)
    
    accuracy_per_section.append( sum(correct)/ len(correct))
    print(per_section)

fig, ax = plt.subplots()
ax.plot(range(1,13), accuracy_per_section, 'g')
ax.fill_between(range(1,13), 0, accuracy_per_section, facecolor='green', alpha=0.2)
ax.set_ylabel('Classification Accuracy', fontsize=12)
ax.set_xlabel('Number of Classes', fontsize=12)
ax.set_xlim(1,12)
ax.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig('/Users/wassy/Documents/fall_2018/am205/final_project/github/wassy_pictures/num_classes_2dpca', dpi=500)

