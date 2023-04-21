# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:08:07 2023

This file hosts all of the user-defined functions referenced in the
"Feature_Labeling_GUI.py" code. 

@author: gaurr
"""

## Importing necessary packages


import numpy as np
import glob
from PIL import Image, ImageDraw, ImageGrab
import io
import base64
import csv
import pandas as pd
import os.path
import PySimpleGUI as sg
import matplotlib
# %matplotlib tk ## will work as a command if file as .ipy extension
# export MPLBACKEND = TKAgg
import matplotlib.pyplot as plt
import cv2 as cv
# matplotlib.use("TkAgg") ##importing matplotlib with Tkinter rather than another framework like Qt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import Feature_Labeling_Variables as var

#%%

# class GUI():
    
#     def __init__(self):
#         ctypes.windll.user32.SetProcessDPIAware()


## Converting JPG to PNG
def jpg_to_png(jpg_path):
    
   """ This function can be called to convert .jpg images to .png format. 
   Args:
       jpg_path (str): The path of the .jpg file to be converted specified with the extension.""" 
   
   # Open the .jpg file using PIL's Image module
   with Image.open(jpg_path) as img:
    # Convert the image to RGBA mode (necessary for saving as .png format)
    img_rgba = img.convert("RGBA")
    
    # Get the file name and extension of the .jpg file
    file_name, file_ext = jpg_path.split("/")[-1].split(".")
    
    # Save the converted image as .png format to the specified directory
    png_dir = os.path.dirname(jpg_path)
    png_path = rf"{file_name}.png"
    img_rgba.save(png_path, format="png")



def parse_folder(window, values):
    
    """ Called when the user selects a folder. This function finds all of the files with 
    .png or .gif extensions in the folder and adds them to the listbox."""
    
    folder = values["-FOLDER-"]
    try:
        # To get the list of files in the folder
        file_list = os.listdir(folder)
    except:
        file_list = []
    
    fnames = [ ## filter down list of files to files ending with extension ".png" or ".gif"
        f
        for f in file_list
        ## case sensitive
        if os.path.isfile(os.path.join(folder, f)) 
        and f.lower().endswith((".gif", ".png"))
    ]
    window["-FILE LIST-"].update(fnames) ## list of files updated with chosen folder's contents

    return fnames
    

def array_to_data(array):
    
    """ Converting images from array type to data to display in the graph element."""
    
    im = Image.fromarray(array)
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

global filename

## User selected an image to display
def disp_image(window, values, fnames, location): 
    
    """ This function is called when the user selects an image they wish to display.
    This can be either when an image is selected from the listbox or the "Next" 
    or "Previous" buttons are selected. 
    
    The graph element is then updated with the image and the filename is also updated. """
    
    if location == 0:
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
    
        index = np.where(np.array(fnames) == values["-FILE LIST-"][0])[0][0]

    elif location == 1:
        
        index = np.where(np.array(fnames) == os.path.basename(window["-TOUT-"].get()))[0][0]
        filename = os.path.join(
            values["-FOLDER-"], fnames[index+1])
        values["-FILE LIST-"][0] = fnames[index+1]
        
    elif location == -1:
        index = np.where(np.array(fnames) == os.path.basename(window["-TOUT-"].get()))[0][0]
        filename = os.path.join(
            values["-FOLDER-"], fnames[index-1])
        values["-FILE LIST-"][0] = fnames[index-1]
    
    window["-TOUT-"].update(filename) ## update the text with the filename selected

    ## converting image type to array and then array to data
    im = Image.open(filename)
    im_array = np.array(im, dtype=np.uint8)
    data = array_to_data(im_array)
    window["-GRAPH-"].draw_image(data=data, location=(0,var.height))

    with open(filename, 'rb') as f:
        im_bytes = f.read()
    pil_image = Image.open(io.BytesIO(im_bytes))
    
    return im, pil_image, filename, values["-FILE LIST-"][0]



def copy(window):
    
    """"This function can be used to copy the annotated image onto the
    user's clipboard."""
    
    widget = window.find_element_with_focus().widget
    if widget.select_present():
        text = widget.selection_get()
        window.TKroot.clipboard_clear()
        window.TKroot.clipboard_append(text)




def overlay_pts(filename):
    data = np.loadtxt(filename, dtype = str)
    x_coords = data[:,2]
    y_coords = data[:,3]
    pmt_ids = data[:,1]
    # print("The length is",len(x_coords)) ## checking how many points with labels there are
    suffix = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
              '20', '21', '22', '23', '24'] ## want to remove all non-dynode and non-bolt points
    index = []
    for i in range(len(x_coords)):
        if not str(pmt_ids[i]).endswith(tuple(suffix)):
            index.append(i)
        
    x_coords = np.delete(x_coords, index)
    y_coords = np.delete(y_coords, index)
    pmt_ids = np.delete(pmt_ids, index)
    # print("The length is now ", len(x_coords)) ## checking that the extra points are removed
   
    return x_coords, y_coords, pmt_ids


def opencv_overlay(pic_file, coords_file, num, base):
    
    """ This function can be used to open images using OpenCV and overlay the labeled points. This will not
    allow the user to move the points and should only be used to view and save images with labels as .png files."""
    
    image = cv.imread(pic_file)
    x, y, ids = overlay_pts(coords_file)
    img_circle = image.copy()
    cv.namedWindow("Trial2"+str(num), cv.WINDOW_NORMAL)
    img2 = cv.resize(img_circle, (4000,2750)) ## cropped original (4000, 3000) to remove watermark
    for i in range(len(x)):
        if str(ids[i]).endswith('00'):
            cv.circle(img2, (int(x[i]), int(y[i])), radius=0, color=(0,0,255), thickness=10)
        else:
            cv.circle(img2, (int(x[i]), int(y[i])), radius=0, color=(0,255,0), thickness=4)
    
    cv.imshow("Trial2"+str(num), img2)
    cv.imwrite("With_points_"+str(base)+".png", img2)

def save_element_as_file(element, filename):
    """
    Saves any element as an image file.  Element needs to have an underlying Widget available (almost if not all of them do)
    :param element: The element to save
    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif)
    """
    widget = element.Widget
    box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
    grab = ImageGrab.grab(bbox=box)
    grab.save(filename)
    
def write_coords_to_csv(dict_name, filename):
    df=pd.DataFrame.from_dict(dict_name, orient='index')
    df = df.transpose()
    df.to_csv(filename, index=None, mode='w')