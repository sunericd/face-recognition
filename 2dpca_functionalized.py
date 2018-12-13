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
dirs = ["9326871/", "9332898/", "9338446_Star/", "9338454/", "9338462/", "9338489/", "9338497/", "9338519/", "9338527/", "9338543/","9414649/", "9416994/"]


#takes in number of principal components, classes, and  test set, and returns test and train images, and principle components among other things
def run_2dpca(components,num_classes,per_section,dirs, path_dir):
    dirs = dirs[:num_classes]
    leaf_mats = []
    leaf_unchanged = []

    pic_list = []
    test_paths = []
    for k in range(0,len(dirs)):
        temp = os.listdir(path_dir + dirs[k])
        random.shuffle(temp)
        len(temp)
        for j in range(0,per_section):
            test_paths.append(dirs[k] +temp[j])
        for ww in range(per_section,len(temp)):
            pic_list.append(dirs[k] +temp[ww])
    #this was for the trying to see which photo an image outside of the test set looks like
    #test_paths.append('9338527/9338527.8.jpg')
        
    train_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(pic_list)/num_classes)) for x in dirs))
    test_class = list(itertools.chain.from_iterable(itertools.repeat(x, int(len(test_paths)/num_classes)) for x in dirs))
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

        #calculates the feature matrix
def make_feature_matrix(sample_image,max_vector, d):
    feature_matrix = []
    for i in range(0,d):
        sample_image.astype(float)        
        feature_matrix.append(np.matmul(sample_image,max_vector[:,i]))
    return(np.array(feature_matrix))

#calculates distances between feature matrix
def distance_function(samp1, samp2,d):
    total_distance = 0 
    feat1 = make_feature_matrix( samp1 , max_vector,d)
    feat2 = make_feature_matrix( samp2 , max_vector,d )
    for l in range(0,d):
        total_distance = total_distance + np.linalg.norm(np.array(feat1[l]) - np.array(feat2[l]))
    return(total_distance)

#determines if nearest neighbor was the correct class
def score(test_set, leaf_mats, train_class,test_class, components,leaf_unchanged,test_unchanged ):
    guess_class = []
    correct = []
    for k in range(0,len(test_set)):        
        distance_to_each = []
        for i in range(0,len(leaf_mats)):
            distance_to_each.append(distance_function(test_set[k], leaf_mats[i] ,components))
        min(distance_to_each)
        min_ind = distance_to_each.index(   min(distance_to_each))
        #print("nearest neighbor:" + str(k) )

        #print("test image" + str(k) )
        guess_class.append(train_class[min_ind])
        #print(pic_list[min_ind])
        if train_class[min_ind] == test_class[k]:
            correct.append(1)
        else:
            correct.append(0)
    return(sum(correct)/ len(correct))


#graphs the rests
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

#hold all else constand and alter...

#number in test set
accuracy_per_section= []
for u in range(2,19 ):
    max_vector, train_class, test_class, leaf_mats, test_set  ,leaf_unchanged, test_unchanged ,avg_face =  run_2dpca( 3, 12, u, dirs, path_dir )
    accuracy_per_section.append( score(test_set, leaf_mats, train_class,test_class, 3, leaf_unchanged, test_unchanged  ) )
graph_results(   range(2,19 ), accuracy_per_section, "test_function_test_set_redo", 'red', "Test Set Size", 2, 18) 
    # Run 1D PCA --> get PCs, eigvals, average face across all classes, average faces per class, test images


#number of classes
accuracy_per_section= []
for u in range(1,13 ):
    max_vector, train_class, test_class, leaf_mats, test_set, leaf_unchanged, test_unchanged ,avg_face =  run_2dpca( 3, u, 4, dirs, path_dir )
    accuracy_per_section.append( score(test_set, leaf_mats, train_class,test_class, 3, leaf_unchanged, test_unchanged  ) )
graph_results(   range(1,13), accuracy_per_section, "test_function_classes_redo", 'green', "Number of Classes" , 1,12) 
     

#components
accuracy_per_section= []
for u in range(0,14 ):
    max_vector, train_class, test_class, leaf_mats, test_set ,leaf_unchanged,test_unchanged ,avg_face=  run_2dpca( u, 12, 4, dirs, path_dir )
    accuracy_per_section.append( score(test_set, leaf_mats, train_class,test_class, u, leaf_unchanged, test_unchanged  ) )
graph_results(   range(0,14 ), accuracy_per_section, "test_function_components_redo",'blue', "Number of Projection Vectors, p", 0 ,13) 

#worst case scenario
max_vector, train_class, test_class, leaf_mats, test_set, leaf_unchanged, test_unchanged,avg_face =  run_2dpca( 1, 12, 19, dirs, path_dir )
print( score(test_set, leaf_mats, train_class,test_class, 1, leaf_unchanged, test_unchanged ) )

#benchmark
max_vector, train_class, test_class, leaf_mats, test_set, leaf_unchanged, test_unchanged,avg_face =  run_2dpca( 3, 12, 4, dirs, path_dir )
print( score(test_set, leaf_mats, train_class,test_class, 3, leaf_unchanged, test_unchanged ) )



#attempt to show the avg matrix
#
#
#avg_for_show= np.empty((200,180,3))
#for i in range(200):
#    for j in range(180):
#        for k in range(3):
#            avg_for_show[i,j,k] = avg_face[ i+k *3 ,j ]
#
#plt.imshow(np.subtract(1,avg_for_show))

