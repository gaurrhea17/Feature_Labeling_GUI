# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:24:03 2023

@author: gaurr
"""

## Importing necessary packages

import numpy as np
import glob
from PIL import Image, ImageDraw, ImageGrab
import io
import csv
import pandas as pd
import os.path
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import ast

import Feature_Labeling_Functions as func
import Feature_Labeling_Variables as var

import sys
sys.path.append(r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Reconstruction\PhotogrammetryAnalysis-master')
# import SK_ring-relabelling-secondattempt as recon

#%% Defining window objects and layout

# sg.theme_previewer() ## Use to see all the colour themes available
sg.theme('GreenTan') ## setting window colour scheme
sg.set_options(dpi_awareness=True)

## Dictionary with necessary lists of coordinates
coord_dict = {"Img": [], "FN": [], "ID": [], "X":[], "Y":[]}

## Top menubar options
menu_butts = [['File', ['New', 'Open', 'Save', 'Exit', ]], ['Edit', ['Copy', '&Undo point', 'Resize Image', 'Change Canvas Size'], ],  ['Help', 'About...'], ]
menubar = [[sg.Menu(menu_butts)],]

## Window column 1: list of files in chosen folder

file_list_col = [
    [
     sg.Text("Image folder"),
     sg.In(size=(25,1), enable_events =True, key="-FOLDER-"),
     sg.FolderBrowse(),
     ],
    [sg.Listbox(values=[], enable_events=True, size=(40,20), key="-FILE LIST-")
     ], ## displays a list of paths to the images you can choose from to display
    ]


mov_col = [[sg.T('Choose what you want to do:', enable_events=True)],
           # [sg.R('Draw Rectangles', 1, key='-RECT-', enable_events=True)],
           # [sg.R('Draw Circle', 1, key='-CIRCLE-', enable_events=True)],
           # [sg.R('Draw Line', 1, key='-LINE-', enable_events=True)],
           [sg.R('Draw PMT points', 1,  key='-PMT_POINT-', enable_events=True)],
           [sg.R('Draw bolt points', 1,  key='-BOLT_POINT-', enable_events=True)],
           [sg.R('Erase item', 1, key='-ERASE-', enable_events=True)],
           [sg.R('Erase all', 1, key='-CLEAR-', enable_events=True)],
           [sg.R('Send to back', 1, key='-BACK-', enable_events=True)],
           [sg.R('Bring to front', 1, key='-FRONT-', enable_events=True)],
           [sg.R('Move Everything', 1, key='-MOVEALL-', enable_events=True)],
           [sg.R('Move Stuff', 1, key='-MOVE-', enable_events=True)],
           [sg.R('Fill in remaining labels', 1, key='-AUTO_LABEL-', enable_events=True)],
           ]

column = [[sg.Graph(canvas_size = (var.width, var.height), graph_bottom_left = (0,0), graph_top_right = (var.width,var.height), 
key = "-GRAPH-", ## can include "change_submits = True"
background_color = 'white', expand_x =True, expand_y = True, enable_events = True, drag_submits = True, right_click_menu=[[],['Erase Item',]])]]

image_viewer_col_2 = [
    [sg.Text("Choose an image from list on the left: ")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Column(column, size=(var.column_width, var.column_height), scrollable = True, key = "-COL-")],
    [sg.Text(key="-INFO-", size=(90,1))],
    ]

post_process_col= [
    [sg.Column(mov_col)],
    [sg.Button("Save Annotations", size = (15,1), key="-SAVE-")],
    [sg.Button("Write CSV", size = (15,1), key="-CSV-")],
    [sg.Button("Reconstruct", size=(15,1), key="-RECON-")],
    # [sg.Slider(range=(1,10), orientation='h', resolution=.1, default_value=1, key='-ZOOM-', enable_events=True),],
    [sg.Button('Zoom In'), sg.Button('Zoom Out')],
    [
     sg.Text("Choose a file of overlay points: "),
     sg.In(size=(15,1), enable_events =True, key="-OVERLAY-"),
     sg.FileBrowse(),
     ],
    ]


## Main function
def main():
    
    # ------ Full Window Layout
    layout= [
        [menubar, sg.Column(file_list_col), sg.VSeperator(), sg.Column(image_viewer_col_2), sg.VSeperator(), sg.Column(post_process_col),
         [sg.Button("Prev", size=(10,1), key="-PREV-"), sg.Button("Next", size=(10,1), key="-NEXT-")],]] 
    
    window = sg.Window("Image Labeling GUI", layout, resizable=True) ## putting together the user interface
    
    location = 0
    
    graph = window["-GRAPH-"] 

    dragging = False
    start_pt = end_pt = prior_rect = None
    ids = None ## used to specify which figure on sg.Graph to delete
    buffer_list = [] ## list to hold coordinates for points being dragged
    
    while True:
        event, values = window.read()
        ## 'event' is the key string of whichever element user interacts with
        ## 'values' contains Python dictionary that maps element key to a value
        
        if event == "Exit" or event == sg.WIN_CLOSED: ## end loop if user clicks "Exit" or closes window
            break;
                 
        # Folder name filled in, we'll now make a list of files in the folder
        if event == "-FOLDER-": ## checking if the user has chosen a folder
            fnames = func.parse_folder(window, values)
                        
        elif event == "-FILE LIST-": ## User selected a file from the listbox
            
            if len(values["-FOLDER-"]) == 0 : ## If user selected listbox before browsing folders
                select_folder = sg.popup_ok("Please select a folder first.")
            
            
            elif len(values["-FOLDER-"]) != 0:
                
                if ids is not None:
                    graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                
                image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=0)
                window.refresh()
                window["-COL-"].contents_changed()
                
                ## Tries to overlay points if the image has associated coordinate file
                x_overlay, y_overlay, id_overlay, pts_fname = func.autoload_pts(values, graph, filename, coord_dict["ID"],
                                                                                coord_dict["X"], coord_dict["Y"])
                
                ## ids, x and y coordinates already added to dictionary
                feature_nums = np.arange(1,len(x_overlay))
                coord_dict["FN"].append(feature_nums)
                coord_dict["Img"].append(filename)
                
                if filename not in coord_dict["Img"]:
                    i=1 ## counter for feature number
        
        if event == "-NEXT-":
            
            if ids is not None:
                graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                
            image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=1)
            window.refresh()
            window["-COL-"].contents_changed()
            
            x_overlay, y_overlay, id_overlay, pts_fname = func.autoload_pts(values, graph, filename, coord_dict["ID"],
                                                                            coord_dict["X"], coord_dict["Y"])
            
            feature_nums = np.arange(1,len(x_overlay))
            coord_dict["FN"].append(feature_nums)
            coord_dict["Img"].append(filename)
            
            if filename not in coord_dict["Img"]:
                i=1 ## counter for feature number
        if event == "-PREV-":
            
            if ids is not None:
                graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
            
            image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=-1)
            window.refresh()
            window["-COL-"].contents_changed()
            
            x_overlay, y_overlay, id_overlay, pts_fname = func.autoload_pts(values, graph, filename, coord_dict["ID"],
                                                                            coord_dict["X"], coord_dict["Y"])
            
            feature_nums = np.arange(1,len(x_overlay)+1)
            coord_dict["FN"].append(feature_nums)
            coord_dict["Img"].append(filename)
            
            if filename not in coord_dict["Img"]:
                i=1 ## counter for feature number
        
## =============== Annotation Features ========================            
            
        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='cross')
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr') 
        
        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            buffer_list.append(values["-GRAPH-"])
            if not dragging:
                start_pt = (x, y)
                dragging = True
                drag_figures = graph.get_figures_at_location((x,y))

                print("Drag figures info", drag_figures, len(drag_figures))        
                lastxy = x,y
            else:
                end_pt = (x,y)
            if prior_rect:
                graph.delete_figure(prior_rect)
                
            delta_x, delta_y = x - lastxy[0], y - lastxy[1] 
            
            lastxy = (x, y)
            
            if None not in (start_pt, end_pt):
                if values['-MOVE-']:
                    if len(drag_figures)==1:
                        pass;
                    elif len(drag_figures)>1: ## removing the background image from the tuple of objects that can be dragged
                        for fig in drag_figures[1:]:
                            graph.move_figure(fig, delta_x, delta_y)
                            graph.update()
                # elif values['-RECT-']:
                #     prior_rect = graph.draw_rectangle(start_pt, end_pt,fill_color='green', line_color='red')
                # elif values['-CIRCLE-']:
                #     prior_rect = graph.draw_circle(start_pt, end_pt[0]-start_pt[0], fill_color='red', line_color='green')
                # elif values['-LINE-']:
                #     prior_rect = graph.draw_line(start_pt, end_pt, width=4)
                elif values['-PMT_POINT-']:
                   graph.draw_point((x,y), color = 'red', size=8)
                elif values['-BOLT_POINT-']:
                    graph.draw_point((x,y), color = 'yellow', size =8)
                elif values['-ERASE-']:
                    for fig in drag_figures:
                        graph.delete_figure(fig)
                elif values['-CLEAR-']:
                    graph.erase()
                elif values['-MOVEALL-']:
                    graph.move(delta_x, delta_y)
                elif values['-FRONT-']:
                    for fig in drag_figures:
                        graph.bring_figure_to_front(fig)
                elif values['-BACK-']:
                    for fig in drag_figures:
                        graph.send_figure_to_back(fig)
            window["-INFO-"].update(value=f"Mouse {values['-GRAPH-']}")
        
        
        elif event.endswith('+UP'):
            window["-INFO-"].update(value=f'Made point at ({end_pt[0]}, {end_pt[1]})')
            
            # if start_pt[0] not in coord_dict["X"] and start_pt[1] not in coord_dict["Y"]: ## if the point is not already there, then add it
            #     coord_dict["Img"].append(filename)
            #     print(coord_dict["FN"][-1])
            #     coord_dict["X"].append(start_pt[0])
            #     coord_dict["Y"].append(start_pt[1])
            
            
            ## check for the points previous coordinates in the dictionary. If they exist, update them
            for i in range(len(coord_dict["X"])):
                x_max, x_min = coord_dict["X"][i]+10, coord_dict["X"][i] - 10
                y_max, y_min = coord_dict["Y"][i]+10, coord_dict["Y"][i] - 10
                
                
                ## checks range of points within +/- 10 pixels due to click uncertainty
                if start_pt[0] < x_max and start_pt [0] > x_min and start_pt[1] < y_max and start_pt[1] > y_min:
                                     
                    print("Found point to replace, ", i)
                    coord_dict["X"][i] = end_pt[0]
                    coord_dict["Y"][i] = end_pt[1]
                    print(f"The old coordinates were {start_pt}")
                    print(f"The new coordinates are {end_pt}")
                    break;
                    
            # print(coord_dict)
            
            if values["-PMT_POINT-"]:
                pmt_id = input("Please enter PMT ID.")
                coord_dict["ID"].append(pmt_id)
                
                i+=1
                j=0
            
            if values["-BOLT_POINT-"]:
                
                ## checks which pmt the bolt belongs to and returns ID of the PMT
                ## along with the angle between the dynode and the bolt
                
                pmt_id, theta = func.bolt_labels(coord_dict, end_pt[0], end_pt[1])
                
                ## was for feature counting with all manual labeling
                # i+=1
                # j+=1
                # if j == 25:
                #     j=0
            
            # func.write_coords_to_csv(coord_dict, 'test.csv')
            
            start_pt, end_pt = None, None  # enable making a new point
            dragging = False
            prior_rect = None
            buffer_list = []
            print("Dragging done. Buffer list: ", buffer_list)
            
            
        elif event == 'Erase item':
            if values['-GRAPH-'] != (None, None):
                delete_figure = graph.get_figures_at_location(values['-GRAPH-'])
                if len(delete_figure) == 1:
                    pass;
                if len(delete_figure)>1:
                    for figure in delete_figure[1:]:
                        graph.delete_figure(figure)
        

## =========================== Overlaying known coordinates =======================        

        if event == "-OVERLAY-": ## if we want to overlay known coordinates on an image
            print(values["-OVERLAY-"])
            x_overlay, y_overlay, id_overlay = func.overlay_pts(values["-OVERLAY-"])
            for i in range(len(x_overlay)):
                if str(id_overlay[i]).endswith('00'):
                    id_overlay[i] = graph.draw_point((x_overlay[i], y_overlay[i]), color = 'red', size=10)
                    coord_dict["ID"].append(id_overlay[i])
                    coord_dict["X"].append(x_overlay[i])
                    coord_dict["Y"].append(y_overlay[i])
                else:
                   id_overlay[i] = graph.draw_point((x_overlay[i], y_overlay[i]), color = 'yellow', size = 8)
                   coord_dict["ID"].append(id_overlay[i])
                   coord_dict["X"].append(x_overlay[i])
                   coord_dict["Y"].append(y_overlay[i])
        
## ======================== Menubar functions =======================================        
        
        elif event == 'Copy':
            func.copy(window)
        
        elif event == '-SAVE-': ## saves annotated image and objects
            dir_name = os.path.dirname(filename)
            base = os.path.basename(filename)
            annotate_fname = str(dir_name)+r"/annotated_"+str(base)
            # func.save_element_as_file(column, annotate_fname)
            
        elif event == '-CSV-':
            func.write_coords_to_csv(coord_dict, filename)
            
        elif event == '&Undo point':
            
            try:
                if len(coord_dict) > 0: 
                    delete_figure = graph.get_figures_at_location(values['-GRAPH-'])
                   
                    for figure in delete_figure[1:]:
                            
                        for k, v in coord_dict.items():
                            print(len(coord_dict))
                            v.pop() ## remove the last point from the dictionary of point coordinates
                            print(len(coord_dict))
                        graph.delete_figure(figure)
            except:
                pass
                    
        elif event == 'Change Canvas Size':
            func.change_canvas_size(window, graph)
            
        elif event == 'Resize Image':
            
            if ids is not None:
                graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
            
            try:
                new_size = ast.literal_eval(input("Please enter the image size '##, ##': "))
                im_array = np.array(image, dtype=np.uint8)
                data = func.resize(im_array, new_size)
                graph.draw_image(data=data, location=(0,var.height))
            except:
                sg.popup_ok("Please check your size input format. E.g. ##, ##")
                
    window.close() ## For when the user presses the Exit button
    
main()


#%% Trying extra programs

## binding mouse click to window image element

# img = cv.imread(filename)
# if img is None:
#     print('Could not read image')
# window["-IMAGE-"].bind("<Button-1>", draw_feature)
    
# base = os.path.basename(filename) ##filename with extension
# img_name = os.path.splitext(base)[0] ##filename without extension
    
# cv.namedWindow(img_name)
# cv.setMouseCallback(img_name, select_feature)
 

# directory =  r'C:\Users\gaurr\TB3\BarrelSurveyRings\images2'
directory = r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Feature Detection & Labelling\Sample_Overlay_SK_Images'
num = 1
for f in os.listdir(directory):
    if f.lower().endswith(".png"):
        f = os.path.join(directory, f)
        pic_file = f
        base = os.path.splitext(os.path.basename(f))[0]
        coords_file =  directory+"/"+base+".txt"
        func.opencv_overlay(pic_file, coords_file, str(num), base)
        num+=1
        if num >= 1:
            break;



## Convert all the BarrelSurveyRings images

# jpg_path = os.path.normpath(r'C:\Users\gaurr\TB3\BarrelSurveyRings\images2')

# func.jpg_folder_to_png(jpg_path)
