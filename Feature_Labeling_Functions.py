# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:08:07 2023

This file hosts all of the user-defined functions referenced in the
"Feature_Labeling_GUI.py" code. 

@author: gaurr
"""

## Importing necessary packages
import sys
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
    filename = os.path.splitext(os.path.basename(jpg_path))[0]
    
    # Save the converted image as .png format to the specified directory
    png_dir = os.path.dirname(jpg_path)
    png_path = os.path.join(png_dir, filename)+".png"
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
            
    

def change_canvas_size(window, graph, scale):
    
    """" This function allows the user to input the size of the graph/column/scrollable area."""
    scale = int(scale)
    graph.Widget.configure(width=var.width*scale, height=var.height*scale)
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
    fnames.sort()
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
    
def get_curr_coords(graph, drag_figures):
    for fig in drag_figures[1:]:
        current_coords = graph.get_bounding_box(fig)
        curr_x = (current_coords[0][0] + current_coords[1][0])/2
        curr_y = (current_coords[0][1] + current_coords[1][1])/2
        point = curr_x, curr_y
        return curr_x, curr_y, point

def resize(window, array, scale):
    
    """ To allow the user to resize the image displayed in the graph."""
    
    im = Image.fromarray(array)
    w, h, c = np.array(array).shape ## width, height, channel
    print(w, h)
    scale = int(scale)
    im = im.resize((w*scale,h*scale), Image.ANTIALIAS)
    print(w*scale, h*scale)
    
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
        
    # window.refresh()
    # window["-COL-"].contents_changed()
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

    with open(filename, 'rb') as f:
        im_bytes = f.read()
    pil_image = Image.open(io.BytesIO(im_bytes))
    
    return im, pil_image, filename, ids



def copy(window):
    
    """"This function can be used to copy the annotated image onto the
    user's clipboard.
    
    DOES NOT CURRENTLY WORK"""
    
    widget = window.find_element_with_focus().widget
    if widget.select_present():
        text = widget.selection_get()
        window.TKroot.clipboard_clear()
        window.TKroot.clipboard_append(text)

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
    except Exception as e:
        sg.popup_ok("The file could not be saved.")
        print(e)

def draw_pts(graph, df, scale=1):

    for index, row in df.iterrows(): 

        ## PMT (dynode) centers
        if str(row[1]).endswith('00'):
            graph.draw_point((row[2]*scale, row[3]*scale), color = 'red', size=10)
            #print("Drew a PMT", row[1], row[2], row[3])

        ## Bolts
        else:
            graph.draw_point((row[2]*scale, row[3]*scale), color = 'yellow', size=8)
            #print("Drew a bolt", row[1], row[2], row[3])

def autoload_pts(filename):
    
    print("Loading existing points file:", filename)
    
    ## Read space delimited point file into dataframe
    df = pd.read_csv(filename, delim_whitespace=True, names=["Img", "ID", "X", "Y", "Name"])
    
    ## Remove points not corresponding to PMT/dynode center nor bolts
    df = df[df['ID'].apply(lambda id: float(id[-2:])<=var.NBOLTS)]    
    
    ## Add columns for automatic bolt labeling variables
    df['R'] = np.nan
    df['theta'] = np.nan
            
    ## Invert Y coordinate from the FeatureReco convention
    if var.invert_y:
        df['Y'] = df['Y'].map(lambda Y: var.height-Y)    
    
    ## Process all the points
    for index, row in df.iterrows(): 

        ## PMT (dynode) centers
        if str(row[1]).endswith('00'):
            pass
        
        ## Bolts
        else:
            ## Get PMT ID associated to this bolt                
            PMT_ID = row[1][:-2] + '00'
            df_pmt = df.loc[df['ID'] == PMT_ID]

            if len(df_pmt.index)>1:
                sys.exit("Unexpected duplicate PMT_ID " + PMT_ID)
            
            ## Calculate parameters needed for automatic bolt labeling
            pmt_x = float(df_pmt['X'].iloc[0])
            pmt_y = float(df_pmt['Y'].iloc[0])
            df.at[index, 'R'] = np.sqrt( (float(row[2])-pmt_x)**2 + (float(row[3])-pmt_y)**2 )
            df.at[index, 'theta'] = angle_to( (float(row[2]), float(row[3])), (pmt_x, pmt_y) )

    return df

def del_point(df, x, y):

    df_feature = df[ (df['X'] == x) & (df['Y'] == y) ]
    
    if len(df_feature.index) == 0:
        print("No feature found at this location!")
        raise

    df.drop(df_feature.index, inplace=True)

    return df_feature

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w+', newline='')

def reverseEnum(data: list):
    for i in range(len(data)-1, -1, -1):
        yield (i, data[i])
    
def write_coords_to_csv(df, filename):

    df = df.sort_values(by=['ID'])
    
    headers=["Img", "ID", "X", "Y", "Name"]
    df[headers].to_csv(filename, index=False, header=False, sep='\t')

    print("Saved Annotations:", filename)

def angle_to(p1, p2, rotation = 270, clockwise=False):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) - rotation
    if not clockwise:
        angle = -angle
    return angle % 360

def get_closest_pmt(df, x, y):
    df_pmts = df[df['ID'].apply(lambda id: float(id[-2:]) == 0)]
    df_closest = df_pmts.iloc[ ((df_pmts['X'] - x)**2 + (df_pmts['Y'] - y)**2).argsort()[:1] ]
    pmt_id = df_closest['ID'].iloc[0][:5]
    pmt_x = df_closest['X'].iloc[0]
    pmt_y = df_closest['Y'].iloc[0]
    return pmt_id, pmt_x, pmt_y
    
def make_bolt(df, bolt_x, bolt_y, name):
    
    ## Find which PMT the bolt is closest to
    ### (WARNING: this will not work if the bolt is closer to a different PMT center))
    pmt_id, pmt_x, pmt_y = get_closest_pmt(df, bolt_x, bolt_y)
    #print("PMT number where min. distance between bolt and dynode :", pmt_id)

    ## Get list of existing bolts for this PMT
    df_bolts = df[df['ID'].apply(lambda id: (id[:5] == pmt_id) & (float(id[-2:]) > 0))]
    #print("The number of bolts for this PMT is ", len(df_bolts.index))

    ## Don't add more than 24 bolts
    if len(df_bolts.index) >= var.NBOLTS:
        raise Exception("Already reached max number of bolts for this PMT! Erase a bolt first.")
    
    ## calculate angle between PMT center and bolt
    bolt_to_pmt = (bolt_x - pmt_x, bolt_y - pmt_y)
    bolt_r = np.sqrt(bolt_to_pmt[0]**2 + bolt_to_pmt[1]**2)
    theta = angle_to((bolt_x, bolt_y), (pmt_x, pmt_y))
    #print(f"Angle between PMT and bolt {theta}")
    
    # Get entry in df_bolts with value closest to 'theta'
    df_closest_theta = df_bolts.iloc[ (abs((theta - df_bolts['theta'] + 180)%360-180)).argsort()[:1] ]            
    #print("Closest theta is ", df_closest_theta)

    # Assign 'ID' +1 if theta is greater than the closest theta or 'ID' -1 if theta is less than the closest theta
    if len(df_closest_theta.index) == 0:
        #print("No bolts for this PMT yet!")
        bolt_label = '-01'

    else:
        closest_bolt_id = int(df_closest_theta['ID'].iloc[0][-2:])        
        closest_theta = df_closest_theta['theta'].iloc[0]        

        # Determine if theta is clockwise to closest_theta
        if (theta - closest_theta - 180)%360 - 180 > 0: # clockwise
            
            if closest_bolt_id == var.NBOLTS:
                bolt_label = '-01'
            
            else:
                bolt_label = '-{:02d}'.format(closest_bolt_id+1)

        # Counter-clockwise
        else:
            
            if closest_bolt_id == 1:
                bolt_label = '-{:02d}'.format(var.NBOLTS)
            
            else:   
                bolt_label = '-{:02d}'.format(closest_bolt_id-1)

    New_ID = pmt_id.zfill(5) + bolt_label

    # append new row to dataframe
    df_new_bolt = pd.DataFrame({'Img': df['Img'].iloc[0], 'ID': New_ID, 'X': bolt_x, 'Y': bolt_y, 'Name': name, 'R': bolt_r, 'theta': theta}, index=[0])
    df = pd.concat([df, df_new_bolt], ignore_index=True)
    #df.loc[len(df.index)] = [df['Img'].iloc[0], New_ID, bolt_x, bolt_y, name, bolt_r, theta]

    print("Inserted your new bolt", df.tail(1))

    # Re-sort dataframe by 'ID'
    #df = df.sort_values(by=['ID'])

    return df

def make_pmt(df, pmt_id, pmt_x, pmt_y, name):
    df_new_pmt = pd.DataFrame({'Img': df['Img'].iloc[0], 'ID': pmt_id.zfill(5)+"-00", 'X': pmt_x, 'Y': pmt_y, 'Name': name, 'R': np.nan, 'theta': np.nan}, index=[0])
    df = pd.concat([df, df_new_pmt], ignore_index=True)

    #df = df.sort_values(by=['ID'])
    print("Inserted your new PMT", df.tail(1))
    
    # recalculate all bolts for this PMT
    #df_bolts = df[df['ID'].apply(lambda id: (id[:5] == pmt_id) & (float(id[-2:]) > 0))]
    
    return df

def move_feature(df, start_pt, end_pt, name):

    # Delete the feature at the start point
    df_feature = del_point(df, start_pt[0], start_pt[1])
    feature_ID = df_feature['ID'].iloc[0]
    #print("Feature being moved is ", feature_ID)

    # If the feature being moved is a PMT
    if str(feature_ID).endswith('00'):
        df = make_pmt(df, feature_ID[:5], end_pt[0], end_pt[1], name)
    
    # Bolt    
    else:
        df = make_bolt(df, end_pt[0], end_pt[1], name)
    
    return df

def plot_labels(graph, df):

    labels = []

    # Loop over all points in the dataframe
    for index, row in df.iterrows(): 

        pmt_id = row[1][:5]
        bolt_id = row[1][-2:]
        draw_x = row[2]
        draw_y = row[3]
        color = 'red' if bolt_id == '00' else 'yellow'
        text = pmt_id if bolt_id == '00' else bolt_id
        
        labels.append(graph.DrawText(text=text, location=(draw_x-10, draw_y-10), color=color))
        
    return labels
      
        
def erase_labels(graph, labels):

    for label in labels:
        graph.delete_figure(label)
        
def get_marker_center(graph, fig):
    current_coords = graph.get_bounding_box(fig)
    curr_x = (current_coords[0][0] + current_coords[1][0])/2
    curr_y = (current_coords[0][1] + current_coords[1][1])/2
    return curr_x, curr_y
    

#%%  
    # for i in range(len(dict_name["ID"])):
    #     for j in range(len(suffix)):
    #         check_name = str(pmt_id)+'-'+suffix[j]
    #         if dict_name["ID"][i].endswith(check_name):
    #             print("Feature right before this bolt, ", dict_name["ID"][i])
    #             check_angle = math.degrees(np.arctan(float(dict_name["X"][i])/float(dict_name["Y"][i])))
    #             print("Angle to check against is, ", check_angle)
                
    #             if theta < check_angle: ## if angle is less than that of the bolt being checked, 
    #                 bolt_label = dict_name["ID"][i].replace(suffix[j],"{:02d}".format(int(suffix[j])-1))
    #                 dict_name["ID"].insert(i+1, bolt_label)
    #                 print("New bolt ID is", dict_name["ID"][i-1])
    #                 print("new bolt: ", bolt_label)
                    
    #                 dict_name["X"].insert(i+1, bolt_x)
    #                 dict_name["Y"].insert(i+1, bolt_y)
    #                 print("New length of dictionary, ", len(dict_name["ID"]))
                    
    #                 return pmt_id, theta
                
    #             if theta > check_angle:
    #                 bolt_label = dict_name["ID"][i].replace(suffix[j],"{:02d}".format(int(suffix[j])+1))
    #                 dict_name["ID"].insert(i+1, bolt_label)
    #                 print("New bolt ID is", dict_name["ID"][i+1])
    #                 print("new bolt: ", bolt_label)
                    
    #                 dict_name["X"].insert(i+1, bolt_x)
    #                 dict_name["Y"].insert(i+1, bolt_y)
    #                 print("New length of dictionary, ", len(dict_name["ID"]))
                    
                    
    #                 return pmt_id, theta
            