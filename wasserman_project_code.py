#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:04:39 2018

@author: wassy
"""
import matplotlib.image as mpimg
import time
from scipy import special
import matplotlib.pyplot as plt
import numpy as np 
import scipy.linalg
import math
import pandas as pd
import itertools
from scipy.interpolate import *
from heapq import nsmallest


start_photo = 1
number_photos = 3
test_start = 11
test_end =  14
path = '/Users/wassy/Documents/fall_2018/am205/final_project/pictures/'

nums = [ i for i in range(start_photo,number_photos+1)]


for i in range(start_photo, len(nums)+1):
    img=mpimg.imread(path + 'sun/s' + str(nums[i-1])+ '.jpg')
    shape = img.shape
    plt.imshow(img)
    flat_arr = img.ravel()
    vector_s = 1 - flat_arr
    if i ==1:
        avg_vec = vector_s
    if i > 1:
        avg_vec = vector_s + avg_vec    

avg_vec = avg_vec/len(nums)
avg_vec_one= 1-avg_vec
arr2 = np.asarray(avg_vec_one).reshape(shape)
plt.imshow(1-(arr2* 255).astype(np.uint8))

