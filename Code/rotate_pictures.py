#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:20:12 2018

@author: wassy
"""

import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

 
path_dir = '/Users/wassy/Documents/fall_2018/am205/final_project/github/faces94/male/9326871/9326871.1.jpg'

#rotatation where person is cut off
def rotate_cut_off(path, angle):
    image = cv2.imread(path)
    rotated = imutils.rotate(image, angle)
    return(rotated)

 
pic1 = rotate_cut_off(path_dir,15)

#rotation where person does not get cut off
def rotate_whole_picture(path, angle):
    image = cv2.imread(path )
    rotated = imutils.rotate_bound(image, angle)
    return(rotated)
    
pic2 = rotate_whole_picture(path_dir,-15)
    
plt.imshow( pic1)
plt.imshow(pic2)

