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

import Feature_Labeling_Functions as func
import Feature_Labeling_Variables as var

import sys
sys.path.append(r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Reconstruction\PhotogrammetryAnalysis-master')
import SK_ring-relabelling-secondattempt as recon

#%% Defining window objects and layout

# sg.theme_previewer() ## Use to see all the colour themes available
sg.theme('GreenTan') ## setting window colour scheme

## Dictionary with necessary lists of coordinates
coord_dict = {"Img": [], "FN": [], "ID": [], "X":[], "Y":[]}

## Top menubar options
menu_butts = [['File', ['New', 'Open', 'Save', 'Exit', ]], ['Edit', ['Cut', 'Copy', 'Paste', 'Undo'], ],  ['Help', 'About...'], ]
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

column = [[sg.Graph(canvas_size = (var.width, var.height), graph_bottom_left = (0,0), 
graph_top_right = (var.width, var.height), key = "-GRAPH-", ## can include "change_submits = True"
background_color = 'white', enable_events = True, drag_submits = True, right_click_menu=[[],['Erase Item',]])]]

image_viewer_col_2 = [
    [sg.Text("Choose an image from list on the left: ")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Column(column, size=(500, 375), scrollable = True, key = "-COL-")],
    [sg.Text(key="-INFO-", size=(60,1))],
    ]

post_process_col= [
    [sg.Column(mov_col)],
    [sg.Button("Save Annotations", size = (15,1), key="-SAVE-")],
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
    
    # Initializing zoom and panning
    zoom_level = 1.0
    pan_position = (0,0)

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
            if len(values["-FOLDER-"]) == 0 :
                select_folder = sg.popup_ok("Please select a folder first.")
            elif len(values["-FOLDER-"]) != 0:
                image, pil_image, filename, csv_file = func.disp_image(window, values, fnames, location=0)
                window.refresh()
                window["-COL-"].contents_changed()
                if filename not in coord_dict["Img"]:
                    i=1 ## counter for feature number
        if event == "-NEXT-":
            image, pil_image, filename, csv_file = func.disp_image(window, values, fnames, location=1)
            window.refresh()
            window["-COL-"].contents_changed()
            if filename not in coord_dict["Img"]:
                i=1 ## counter for feature number
        if event == "-PREV-":
            image, pil_image, filename, csv_file = func.disp_image(window, values, fnames, location=-1)
            window.refresh()
            window["-COL-"].contents_changed()
            if filename not in coord_dict["Img"]:
                i=1 ## counter for feature number
        
## =============== Annotation Features ========================            
            
        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='fleur')
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr') 
        
        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            if not dragging:
                start_pt = (x, y)
                dragging = True
                drag_figures = graph.get_figures_at_location((x,y))[-1]
                # print("Drag figures info", drag_figures, len(drag_figures))
                lastxy = x,y
            else:
                end_pt = (x,y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = (x, y)
            
            if None not in (start_pt, end_pt):
                if values['-MOVE-']:
                    # for fig in drag_figures:
                    #     graph.move_figure(fig, delta_x, delta_y)
                    #     graph.update()
                    graph.move_figure(drag_figures, delta_x, delta_y)
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
                # elif values['-FRONT-']:
                #     for fig in drag_figures:
                #         graph.bring_figure_to_front(fig)
                # elif values['-BACK-']:
                #     for fig in drag_figures:
                #         graph.send_figure_to_back(fig)
            window["-INFO-"].update(value=f"Mouse {values['-GRAPH-']}")
        
        elif values["-PMT_POINT-"] and event.endswith('+UP'):  # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"Made point at ({start_pt[0]}, {start_pt[1]})")
            
            coord_dict["Img"].append(filename)
            coord_dict["FN"].append(i)
            coord_dict["X"].append(start_pt[0])
            coord_dict["Y"].append(start_pt[1])
            ## == insert function to put in PMT ID
            pmt_id = input("Please enter PMT ID.")
            coord_dict["ID"].append(pmt_id)
            
            start_pt, end_pt = None, None  # enable making a new point
            dragging = False
            prior_rect = None
            i+=1
            j=0
            
        elif values["-BOLT_POINT-"] and event.endswith('+UP'):  # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"Made point at ({start_pt[0]}, {start_pt[1]})")
            
            coord_dict["Img"].append(filename)
            coord_dict["FN"].append(i)
            coord_dict["X"].append(start_pt[0])
            coord_dict["Y"].append(start_pt[1])
            ## == insert function to put in PMT ID
            coord_dict["ID"].append(str(pmt_id+"-0"+str(j)))
            
            start_pt, end_pt = None, None  # enable making a new point
            dragging = False
            prior_rect = None
            i+=1
            j+=1
            if j == 25:
                j=0
            # event == "-COL-"
            
        elif event.endswith('+RIGHT+'):  # Righ click
            window["-INFO-"].update(value=f"Right clicked location {values['-GRAPH-']}")
        elif event.endswith('+MOTION+'):  # Righ click
            window["-INFO-"].update(value=f"mouse freely moving {values['-GRAPH-']}")
        elif event == 'Erase item':
            if values['-GRAPH-'] != (None, None):
                drag_figures = graph.get_figures_at_location(values['-GRAPH-'])
                for figure in drag_figures:
                    graph.delete_figure(figure)
                    
## =========================== Zooming and Panning ================================

        # # Handle image zoom and pan
        # if event == '-ZOOM-':
        #     zoom_level = values['-ZOOM-']
        #     if image:
        #         zoomed_image = zoom_image(image_paths[image_index], zoom_level, pan_position)
        #         window['image_display'].update(data=cv2.imencode('.png', zoomed_image)[1].tobytes())
        
        # if event == 'image_display':
        #     if values['image_display']:
        #         x, y = values['image_display']
        #         if zoom_level != 1.0:
        #             pan_position[0] += int(x / zoom_level) - int((pan_position[0] + x) / zoom_level)
        #             pan_position[1] += int(y / zoom_level) - int((pan_position[1] + y) / zoom_level)
        #             zoomed_image = zoom_image(image_paths[image_index], zoom_level, pan_position)
        #             window['image_display'].update(data=cv2.imencode('.png', zoomed_image)[1].tobytes())
        
## either open up new CV 
        
        # if event == 'Zoom In':
        #     zoom_level *= 1.2
        #     new_width = int(pil_image.width * zoom_level)
        #     new_height = int(pil_image.height * zoom_level)
        #     resized_image = pil_image.resize((new_width, new_height))
        #     photo_image = sg.tkinter.PhotoImage(resized_image)
        #     graph.erase()
        #     graph.draw_image(data=photo_image, location=(0, var.height))
            
        # if event == 'Zoom Out':
        #     zoom_level /= 1.2
        #     new_width = int(pil_image.width * zoom_level)
        #     new_height = int(pil_image.height * zoom_level)
        #     resized_image = pil_image.resize((new_width, new_height))
        #     photo_image = sg.tkinter.PhotoImage(resized_image)
        #     graph.erase()
        #     graph.draw_image(data=photo_image, location=(0, var.height))
            
        # Pan around the image
        # if event == '-GRAPH-':
        #     x, y = values['-GRAPH-']
        #     dx = x - pan_position[0]
        #     dy = y - pan_position[1]
        #     pan_position = (x, y)
        #     graph.move(-dx, dy)

## =========================== Overlaying known coordinates =======================        

        if event == "-OVERLAY-": ## if we want to overlay known coordinates on an image
            print(values["-OVERLAY-"])
            x_overlay, y_overlay, pmt_id_overlay = func.overlay_pts(values["-OVERLAY-"])
            for i in range(len(x_overlay)):
                # print(x_overlay[i], y_overlay[i])
                graph.draw_point((int(x_overlay[i]), int(y_overlay[i])), size=8)
        
        elif event == 'Copy':
            func.copy(window)
        
        elif event == '-SAVE-':
            dir_name = os.path.dirname(filename)
            base = os.path.basename(filename)
            annotate_fname = str(dir_name)+r"/annotated_"+str(base)
            func.save_element_as_file(graph, annotate_fname)
            
        elif event == 'Undo':
            if len(coord_dict) > 0:
                for k, v in coord_dict.items():
                    v.pop() ## remove the last point from the dictionary of point coordinates
                
                ## redraw the image on the graph element
                image, pil_image, filename, value_file = func.disp_image(window, values, fnames, location=1)
                
                ## draw the associated points on the image again? 
                for i in range(len(coord_dict)):
                    graph.draw_point((coord_dict["X"][i], coord_dict["Y"][i]), color = 'red', size=8)
                
    window.close() ## For when the user presses the Exit button
    
main()

print(coord_dict)

# func.write_coords_to_csv(coord_dict, 'test.csv')

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
 


# num = 1
# for f in os.listdir(directory):
#     if f.lower().endswith(".png"):
#         f = os.path.join(directory, f)
#         pic_file = f
#         base = os.path.splitext(os.path.basename(f))[0]
#         coords_file = os.path.dirname(f)+"/"+base+".txt"
#         func.opencv_overlay(pic_file, coords_file, num, base)
#         num+=1
