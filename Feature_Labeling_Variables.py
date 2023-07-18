# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:20:21 2023

This file holds any/all global variables module_namespaces required in 
both the functions and GUI modules for the feature labeling GUI.

@author: gaurr
"""

import numpy as np

width, height = 4000, 2750 ## sizes of images
column_width = 1000
column_height = 750
invert_y = True
NBOLTS = 24

row_col_bound = 200

## model values obtained from MATLAB for the old drone calibration
fx, fy, cx, cy, k1, k2, k3, p1, p2, skew = 3110.7, 3112.8, 1988.2, 1510.5, -0.2497, 0.1693, 0, -9.7426e-4, -8.7611e-4, -6.8332

mtx = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])

# Convert distortion coefficients to numpy array
dist = np.array([k1, k2, p1, p2])

global value_file, graph