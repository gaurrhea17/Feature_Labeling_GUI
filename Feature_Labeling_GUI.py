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
    [sg.Button("Plot Labels", size = (15,1), key="-PLOT_LABEL-"), sg.Button("Remove Labels", size = (15,1), key="-ERASE_LABEL-")],
    [sg.Button("Write CSV", size = (15,1), key="-CSV-")],
    [sg.Button("Reconstruct", size=(15,1), key="-RECON-")],
    # [sg.Slider(range=(1,10), orientation='h', resolution=.1, default_value=1, key='-ZOOM-', enable_events=True),],
    [sg.Button('Zoom In'), sg.Button('Zoom Out')],
    [sg.Button("Shift R"), sg.Button("Shift L")],
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
    
    name = input("Please enter your initials. ")

    dragging = False
    start_pt = end_pt = None
    ids = None ## used to specify which figure on sg.Graph to delete
        
    while True:
        event, values = window.read(timeout=100)
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
                    # graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                    graph.erase()
                    
                image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=0)
                window.refresh()
                window["-COL-"].contents_changed()
                
                ## Tries to overlay points if the image has associated coordinate file
                x_overlay, y_overlay, id_overlay, pts_fname, coord_dict = func.autoload_pts(values, graph, filename, name)
                print("Length of polar coordinate lists", len(coord_dict["R"]))
                ## ids, x and y coordinates already added to dictionary
                ## can add feature number here but for now, no use
                coord_dict["Img"].append(filename)
                
                if filename not in coord_dict["Img"]:
                    i=1 ## counter for feature number
        
        if event == "-NEXT-":
            
            if ids is not None:
                # graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                graph.erase()
                
            image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=1)
            window.refresh()
            window["-COL-"].contents_changed()
            
            x_overlay, y_overlay, id_overlay, pts_fname, coord_dict = func.autoload_pts(values, graph, filename, name)
            
            coord_dict["Img"].append(filename)
            coord_dict["Name"].append(input("Please enter your initials."))
            
            if filename not in coord_dict["Img"]:
                i=1 ## counter for feature number
        if event == "-PREV-":
            
            if ids is not None:
                # graph.delete_figure(ids) ## delete the figure on the canvas if displaying a new one
                graph.erase()
                
            image, pil_image, filename, ids = func.disp_image(window, values, fnames, location=-1)
            window.refresh()
            window["-COL-"].contents_changed()
            
            x_overlay, y_overlay, id_overlay, pts_fname, coord_dict = func.autoload_pts(values, graph, filename, name)
            
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
            if not dragging:
                dragging = True
                drag_figures = graph.get_figures_at_location((x,y))

                for fig in drag_figures:
                    current_coords = graph.get_bounding_box(fig)
                    curr_x = (current_coords[0][0] + current_coords[1][0])/2
                    curr_y = (current_coords[0][1] + current_coords[1][1])/2
                    start_pt = curr_x, curr_y
            else:
                end_pt = x, y
                for fig in drag_figures[1:]:
                    current_coords = graph.get_bounding_box(fig)
                    curr_x = (current_coords[0][0] + current_coords[1][0])/2
                    curr_y = (current_coords[0][1] + current_coords[1][1])/2
                    end_pt = curr_x, curr_y
                
            if None not in (start_pt, end_pt):
                if values['-MOVE-']:
                    if len(drag_figures)==1:
                        pass;
                    elif len(drag_figures)>1: ## removing the background image from the tuple of objects that can be dragged
                        for fig in drag_figures[1:]:
                            current_coords = graph.get_bounding_box(fig)
                            curr_x = (current_coords[0][0] + current_coords[1][0])/2
                            curr_y = (current_coords[0][1] + current_coords[1][1])/2
                            graph.move_figure(fig, x - curr_x, y - curr_y)
                            graph.update()
                elif values['-ERASE-']:
                    for fig in drag_figures:
                        graph.delete_figure(fig)
                elif values['-CLEAR-']:
                    graph.erase()
                # elif values['-MOVEALL-']:
                #     graph.move(delta_x, delta_y)
                elif values['-FRONT-']:
                    for fig in drag_figures:
                        graph.bring_figure_to_front(fig)
                elif values['-BACK-']:
                    for fig in drag_figures:
                        graph.send_figure_to_back(fig)
            window["-INFO-"].update(value=f"Mouse {values['-GRAPH-']}")
        
        
        elif event.endswith('+UP') and values['-MOVE-']:
            
            try:
            
                window["-INFO-"].update(value=f"Moved point from {start_pt} to {end_pt}")
                
                for i in range(len(coord_dict["X"])):
                    if coord_dict["X"][i] == start_pt[0] and coord_dict["Y"][i] == start_pt[1]:
                        coord_dict["X"][i] = end_pt[0]
                        coord_dict["Y"][i] = end_pt[1]
                        for j, d in func.reverseEnum(coord_dict["ID"]):
                            if str(d)[:5] == str(coord_dict["ID"][i])[:5] and str(d).endswith('00'):
                                pmt_x, pmt_y = coord_dict["X"][j], coord_dict["Y"][j]
                                coord_dict["R"][i] = np.sqrt((coord_dict["X"][i] - pmt_x)**2 + (coord_dict["Y"][i] - pmt_y)**2)
                                coord_dict["theta"][i] = func.angle_to((pmt_x, pmt_y), (coord_dict["X"][i], coord_dict["Y"][i]))
                                
            except Exception as e:
                print(e)
                    
            start_pt, end_pt = None, None  # enable making a new point
            dragging = False
            
        elif event.endswith('+UP') and not values["-MOVE-"]: ## Clicked and made a point
            
            try:
                window["-INFO-"].update(value=f'Made point at ({end_pt[0]}, {end_pt[1]})')
            except Exception as e:
                print("Did not make point. Please try again.")
                print(e)
            
            
            if values["-PMT_POINT-"]:
                ## drawing PMT point
                graph.draw_point((x,y), color = 'red', size=8)
                
                pmt_id = input("Please enter PMT ID.")
                coord_dict["ID"].append(pmt_id)
                coord_dict["X"].append(end_pt[0])
                coord_dict["Y"].append(end_pt[1])
                coord_dict["Name"].append(name)
                
            
            if values["-BOLT_POINT-"]:
                try: ## drawing bolt point
                    new_bolt = graph.draw_point((x,y), color = 'yellow', size =8)
                    
                    ## checks which pmt the bolt belongs to and returns ID of the PMT
                    ## along with the angle between the dynode and the bolt
                    
                    pmt_id, theta, bolt_label = func.bolt_labels(coord_dict, end_pt[0], end_pt[1], name)
                    print("You just added a bolt ", bolt_label)
                except Exception as e:
                    print(e)
                    print("Your last point could not be added. Please try again.")
                    graph.delete_figure(new_bolt)
                    
            start_pt, end_pt = None, None  # enable making a new point
            dragging = False
            
            
        elif event == 'Erase item':
            if values['-GRAPH-'] != (None, None):
                delete_figure = graph.get_figures_at_location(values['-GRAPH-'])
                if len(delete_figure) == 1:
                    pass;
                if len(delete_figure)>1:
                    for figure in delete_figure[1:]:
                        graph.delete_figure(figure)
        
        elif event.endswith('+MOTION+'):
            window["-INFO-"].update(value=f"Mouse freely moving {values['-GRAPH-']}")

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
            annotate_fname = os.path.join(str(dir_name), "annotated_"+str(base))
            # func.save_element_as_file(column, annotate_fname)
            
        elif event == '-PLOT_LABEL-':
            pmt_labels, bolt_labels = func.plot_labels(coord_dict, graph)
            
        elif event == '-ERASE_LABEL-':
            func.erase_labels(graph, pmt_labels, bolt_labels)
        
        elif event == '-CSV-':
            try:
                func.write_coords_to_csv(coord_dict, filename, values)
                print("Annotations saved!")
            except:
                print("Did not save. Check that the file is not open.")
            
        # elif event == 'Shift R':
            
        
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
            scale = input("Please enter a scale factor '##': ")
            func.change_canvas_size(window, graph, scale)
            
        elif event == 'Resize Image':
            
            if ids is not None:
                # graph.delete_figure(ids) ## removes only the image
                graph.erase() ## removes all figures drawn on the graph
            
            # try:
            scale = input("Please enter a scale factor '##': ")
            im_array = np.array(image, dtype=np.uint8)
            data = func.resize(window, im_array, scale)
            func.change_canvas_size(window, graph, scale)
            graph.draw_image(data=data, location=(0,var.height))
            func.redraw_pts(coord_dict, graph, scale)
            
                
            # except:
            #     sg.popup_ok("Please check your size input format. E.g. ##, ##")
        
    window.refresh() ## refreshing the GUI to prevent it from hanging
    window.close() ## For when the user presses the Exit button
    
main()


#%%


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
