# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:08:07 2023

This file hosts all of the user-defined functions referenced in the
"Feature_Labeling_GUI.py" code. 

@author: gaurr
"""

## Importing necessary packages

import json
import math
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


img_types = (".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG")

## Converting a single JPG to PNG
def jpg_to_png(jpg_path):
    
   """ This function can be called to convert .jpg images to .png format. 
   Args:
       jpg_path (str): The path of the .jpg file to be converted specified with the extension.""" 
   
   # Open the .jpg file using PIL's Image module
   with Image.open(jpg_path) as img:
    # Convert the image to RGBA mode (necessary for saving as .png format)
    img_rgba = img.convert("RGBA")
    
    # Get the file name and extension of the .jpg file
    file_name = os.path.splitext(os.path.basename(jpg_path))[0]
    
    # Save the converted image as .png format to the specified directory
    png_dir = os.path.dirname(jpg_path)
    png_path = rf"{png_dir}\{file_name}.png"
    img_rgba.save(png_path, format="png")
    
    return png_path

## Converting a folder of JPGs to PNGs

def jpg_folder_to_png(directory):
    for fname in os.listdir(directory):
        if fname.endswith(".jpg"):
            im = Image.open(os.path.join(directory, fname))
            img_rgba = im.convert('RGBA')
            
            filename = os.path.splitext(fname)[0]
            new_filename = str(os.path.join(directory, filename)) + '.png'
            img_rgba.save(new_filename)
            
            os.remove(os.path.join(directory, fname)) ## deletes the .jpg
            continue
        else:
            continue
            
    

def change_canvas_size(window, graph):
    
    """" This function allows the user to input the size of the graph/column/scrollable area."""
    
    width, height = input("Please enter a width and then a height with a space ('## ##'): ").split()
    graph.Widget.configure(width=width, height=height)
    window.refresh()
    window["-COL-"].contents_changed()


def parse_folder(window, values):
    
    """ Called when the user selects a folder. This function finds all of the files with 
    .png or .gif extensions in the folder and adds them to the listbox."""
    
    folder = values["-FOLDER-"]
    try:
        # To get the list of files in the folder
        file_list = os.listdir(folder)
    except:
        file_list = []
    
    fnames = [ ## filter down list of files to files ending with extensions from img_types
        f
        for f in file_list
        ## case sensitive
        if os.path.isfile(os.path.join(folder, f)) and 
        f.lower().endswith(img_types)
    ]
    window["-FILE LIST-"].update(fnames) ## list of files updated with chosen folder's contents

    return fnames
    

def array_to_data(im, filename):
    
    """ Converting images from array type to data to display in the graph element."""
    
    # im = Image.fromarray(array)

    with io.BytesIO() as output:
        if im.format == "PNG":
            im.save(output, format="PNG")
        else:
            png_img = jpg_to_png(filename)
            png_img = Image.open(png_img)
            png_img.save(output, format="PNG")
        data = output.getvalue()
    return data


def resize(array, new_size):
    
    """ To allow the user to resize the image displayed in the graph."""
    
    im = Image.fromarray(array)
    im = im.resize(new_size, Image.ANTIALIAS)
    
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
    # im_array = np.array(im, dtype=np.uint8) ##convering to array is unnecessary?
    data = array_to_data(im, filename)
    ids = window["-GRAPH-"].draw_image(data=data, location=(0,var.height))

    ## Open .csv file to write feature coordinates to
    csv_file = open(os.path.splitext(filename)[0]+".csv", "a+")

    with open(filename, 'rb') as f:
        im_bytes = f.read()
    pil_image = Image.open(io.BytesIO(im_bytes))
    
    return im, pil_image, filename, csv_file, ids



def copy(window):
    
    """"This function can be used to copy the annotated image onto the
    user's clipboard.
    
    DOES NOT CURRENTLY WORK"""
    
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


def opencv_overlay(pic_file, coords_file, annotate_fname, base):
    
    """ This function can be used to open images using OpenCV and overlay the labeled points. This will not
    allow the user to move the points and should only be used to view and save images with labels as .png files."""
    
    image = cv.imread(pic_file)
    x, y, ids = overlay_pts(coords_file)
    img_circle = image.copy()
    
    annotate_fname = str(pic_file)
    cv.namedWindow(annotate_fname, cv.WINDOW_NORMAL)
    img2 = cv.resize(img_circle, (4000,2750)) ## cropped original (4000, 3000) to remove watermark
    for i in range(len(x)):
        if str(ids[i]).endswith('00'):
            cv.circle(img2, (int(float(x[i])), int(float(y[i]))), radius=0, color=(0,0,255), thickness=10)
        else:
            cv.circle(img2, (int(float(x[i])), int(float(y[i]))), radius=0, color=(0,255,0), thickness=4)
    
    cv.imshow(annotate_fname, img2)
    # cv.imwrite("With_points_"+str(base)+".png", img2)


def save_element_as_file(element, filename):
    """
    Saves any element as an image file.  Element needs to have an underlying Widget available (almost if not all of them do)
    :param element: The element to save
    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif)
    """
    try:
        widget = element.Widget
        # box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
        box = (0, 0, 500, 500)
        grab = ImageGrab.grab(bbox=box)
        grab.save(filename)
        sg.popup_ok("Your image has been saved with annotations!")
    except:
        sg.popup_ok("The file could not be saved.")


def autoload_pts(graph, filename, id_dict, x_dict, y_dict):
    try:
    
        pts_dir = r'C:\Users\gaurr\TB3\BarrelSurveyRings\points'
        pts_file = os.path.basename(filename).split('.')[0]
        pts_fname = os.path.join(pts_dir, pts_file) + ".txt" 
        x_overlay, y_overlay, id_overlay = overlay_pts(pts_fname) ## overlaying coordinates
        for i in range(len(x_overlay)):
            if str(id_overlay[i]).endswith('00'):
                graph.draw_point((float(x_overlay[i]), 2750 - float(y_overlay[i])), color = 'red', size=10)
                id_dict.append(id_overlay[i])
                x_dict.append(x_overlay[i])
                y_dict.append(y_overlay[i])
            else:
                graph.draw_point((float(x_overlay[i]), 2750 - float(y_overlay[i])), color = 'yellow', size = 8)
                id_dict.append(id_overlay[i])
                x_dict.append(x_overlay[i])
                y_dict.append(y_overlay[i])
        
        return x_overlay, y_overlay, id_overlay, pts_fname
        
    except:
        pass
    
def write_coords_to_csv(dict_name, filename):
    df=pd.DataFrame.from_dict(dict_name, orient='index')
    df = df.transpose()
    df.to_csv(filename, index=None, mode='w')
    
def bolt_labels(dict_name, bolt_x, bolt_y):
    buffer_r = []
    suffix = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
              '20', '21', '22', '23', '24']
    for i in range(len(dict_name["ID"])):
        if str(dict_name["ID"][i]).endswith('00'):
            x = abs(bolt_x - dict_name["X"][i])
            y = abs(bolt_y - dict_name["Y"][i])
            buffer_r.append(np.sqrt(x**2 + y**2))
    
    pmt_id = np.where(buffer_r == min(buffer_r))[0]
    
    theta = math.degrees(np.arctan(x/y)) ## angle between dynode and bolt
    for i in range(len(dict_name["ID"])):
        check_name = str(pmt_id)+'-'+tuple(suffix)
        if str(dict_name["ID"][i]).endswith(check_name):
            check_angle = math.degrees(np.arctan(float(dict_name["X"][i])/float(dict_name["Y"][i])))
            if theta > check_angle:
                for j in range(len(suffix)):
                    if check_name.endswith(int(suffix[j])):
                        dict_name["ID"].append('0000'+pmt_id+'-'+str(int(suffix[j]+1)))
                
                dict_name["X"].append(bolt_x)
                dict_name["Y"].append(bolt_y)
    
    return pmt_id, theta
            